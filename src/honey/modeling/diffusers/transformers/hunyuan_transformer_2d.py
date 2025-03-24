# TODO: sync attention_processor for HunyuanAttnProcessor2_0
# TODO: sync embeddings for HunyuanCombinedTimestepTextSizeStyleEmbedding

from typing import Dict, Optional, Union

from honey.compiler import ops

from honey.frontend import nn, Tensor

from ..attention import FeedForward
from ..attention_processor import (  # , HunyuanAttnProcessor2_0
    Attention,
    AttentionProcessor,
)
from ..embeddings import (
    # HunyuanCombinedTimestepTextSizeStyleEmbedding,
    PatchEmbed,
    PixArtAlphaTextProjection,
)
from ..modeling_outputs import Transformer2DModelOutput

from ..normalization import AdaLayerNormContinuous, FP32LayerNorm


class AdaLayerNormShift(nn.Module):
    r"""
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(
        self,
        embedding_dim: int,
        elementwise_affine=True,
        eps=1e-6,
        dtype: str = "float16",
    ):
        super().__init__()
        self.silu = ops.silu
        self.linear = nn.Linear(embedding_dim, embedding_dim, dtype=dtype)
        self.norm = FP32LayerNorm(
            embedding_dim, elementwise_affine=elementwise_affine, eps=eps
        )

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        shift = self.linear(
            ops.cast()(self.silu(ops.cast()(emb, dtype="float32"))), emb.dtype()
        )
        x = self.norm(x) + ops.unsqueeze(1)(shift)
        return x


class HunyuanDiTBlock(nn.Module):
    r"""
    Transformer block used in Hunyuan-DiT model (https://github.com/Tencent/HunyuanDiT). Allow skip connection and
    QKNorm

    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of headsto use for multi-head attention.
        cross_attention_dim (`int`,*optional*):
            The size of the encoder_hidden_states vector for cross attention.
        dropout(`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        activation_fn (`str`,*optional*, defaults to `"geglu"`):
            Activation function to be used in feed-forward. .
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, *optional*, defaults to 1e-6):
            A small constant added to the denominator in normalization layers to prevent division by zero.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*):
            The size of the hidden layer in the feed-forward block. Defaults to `None`.
        ff_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the feed-forward block.
        skip (`bool`, *optional*, defaults to `False`):
            Whether to use skip connection. Defaults to `False` for down-blocks and mid-blocks.
        qk_norm (`bool`, *optional*, defaults to `True`):
            Whether to use normalization in QK calculation. Defaults to `True`.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        cross_attention_dim: int = 1024,
        dropout=0.0,
        activation_fn: str = "geglu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-6,
        final_dropout: bool = False,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        skip: bool = False,
        qk_norm: bool = True,
        dtype: str = "float16",
    ):
        super().__init__()

        # Define 3 blocks. Each block has its own normalization layer.
        # NOTE: when new version comes, check norm2 and norm 3
        # 1. Self-Attn
        self.norm1 = AdaLayerNormShift(
            dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps, dtype=dtype
        )

        self.attn1 = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=dim // num_attention_heads,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=True,
            processor=HunyuanAttnProcessor2_0(),
            dtype=dtype,
        )

        # 2. Cross-Attn
        self.norm2 = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)

        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            dim_head=dim // num_attention_heads,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=True,
            processor=HunyuanAttnProcessor2_0(),
            dtype=dtype,
        )
        # 3. Feed-forward
        self.norm3 = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)

        self.ff = FeedForward(
            dim,
            dropout=dropout,  ### 0.0
            activation_fn=activation_fn,  ### approx GeLU
            final_dropout=final_dropout,  ### 0.0
            inner_dim=ff_inner_dim,  ### int(dim * mlp_ratio)
            bias=ff_bias,
            dtype=dtype,
        )

        # 4. Skip Connection
        if skip:
            self.skip_norm = FP32LayerNorm(2 * dim, norm_eps, elementwise_affine=True)
            self.skip_linear = nn.Linear(2 * dim, dim, dtype=dtype)
        else:
            self.skip_linear = None

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        temb: Optional[Tensor] = None,
        image_rotary_emb=None,
        skip=None,
    ) -> Tensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Long Skip Connection
        if self.skip_linear is not None:
            cat = ops.concatenate()([hidden_states, skip], dim=-1)
            cat = self.skip_norm(cat)
            hidden_states = self.skip_linear(cat)

        # 1. Self-Attention
        norm_hidden_states = self.norm1(
            hidden_states, temb
        )  ### checked: self.norm1 is correct
        attn_output = self.attn1(
            norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states = hidden_states + attn_output

        # 2. Cross-Attention
        hidden_states = hidden_states + self.attn2(
            self.norm2(hidden_states),
            encoder_hidden_states=encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        # FFN Layer ### TODO: switch norm2 and norm3 in the state dict
        mlp_inputs = self.norm3(hidden_states)
        hidden_states = hidden_states + self.ff(mlp_inputs)

        return hidden_states


class HunyuanDiT2DModel(nn.Module):
    """
    HunYuanDiT: Diffusion model with a Transformer backbone.

    Inherit ModelMixin and ConfigMixin to be compatible with the sampler StableDiffusionPipeline of diffusers.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88):
            The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        patch_size (`int`, *optional*):
            The size of the patch to use for the input.
        activation_fn (`str`, *optional*, defaults to `"geglu"`):
            Activation function to use in feed-forward.
        sample_size (`int`, *optional*):
            The width of the latent images. This is fixed during training since it is used to learn a number of
            position embeddings.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        cross_attention_dim (`int`, *optional*):
            The number of dimension in the clip text embedding.
        hidden_size (`int`, *optional*):
            The size of hidden layer in the conditioning embedding layers.
        num_layers (`int`, *optional*, defaults to 1):
            The number of layers of Transformer blocks to use.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            The ratio of the hidden layer size to the input size.
        learn_sigma (`bool`, *optional*, defaults to `True`):
             Whether to predict variance.
        cross_attention_dim_t5 (`int`, *optional*):
            The number dimensions in t5 text embedding.
        pooled_projection_dim (`int`, *optional*):
            The size of the pooled projection.
        text_len (`int`, *optional*):
            The length of the clip text embedding.
        text_len_t5 (`int`, *optional*):
            The length of the T5 text embedding.
    """

    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        patch_size: Optional[int] = None,
        activation_fn: str = "gelu-approximate",
        sample_size=32,
        hidden_size=1152,
        num_layers: int = 28,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = True,
        cross_attention_dim: int = 1024,
        norm_type: str = "layer_norm",
        cross_attention_dim_t5: int = 2048,
        pooled_projection_dim: int = 1024,
        text_len: int = 77,
        text_len_t5: int = 256,
        dtype: str = "float16",
        **kwargs,
    ):
        super().__init__()
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.num_heads = num_attention_heads
        self.inner_dim = num_attention_heads * attention_head_dim

        self.text_embedder = PixArtAlphaTextProjection(
            in_features=cross_attention_dim_t5,
            hidden_size=cross_attention_dim_t5 * 4,
            out_features=cross_attention_dim,
            act_fn="silu_fp32",
            dtype=dtype,
        )

        self.text_embedding_padding = nn.Parameter(
            [text_len + text_len_t5, cross_attention_dim], dtype=dtype
        )

        self.pos_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
            patch_size=patch_size,
            pos_embed_type=None,
            dtype=dtype,
        )

        self.time_extra_emb = HunyuanCombinedTimestepTextSizeStyleEmbedding(
            hidden_size,
            pooled_projection_dim=pooled_projection_dim,
            seq_len=text_len_t5,
            cross_attention_dim=cross_attention_dim_t5,
            dtype=dtype,
        )

        # HunyuanDiT Blocks
        self.blocks = nn.ModuleList(
            [
                HunyuanDiTBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    activation_fn=activation_fn,
                    ff_inner_dim=int(self.inner_dim * mlp_ratio),
                    cross_attention_dim=cross_attention_dim,
                    qk_norm=True,  # See http://arxiv.org/abs/2302.05442 for details.
                    skip=layer > num_layers // 2,
                    dtype=dtype,
                )
                for layer in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim,
            self.inner_dim,
            elementwise_affine=False,
            eps=1e-6,
            dtype=dtype,
        )
        self.proj_out = nn.Linear(
            self.inner_dim,
            patch_size * patch_size * self.out_channels,
            bias=True,
            dtype=dtype,
        )

    def set_attn_processor(
        self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]
    ):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def forward(
        self,
        hidden_states: Tensor,
        timestep: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        text_embedding_mask: Optional[Tensor] = None,
        encoder_hidden_states_t5: Optional[Tensor] = None,
        text_embedding_mask_t5: Optional[Tensor] = None,
        image_meta_size: Optional[Tensor] = None,
        style: Optional[Tensor] = None,
        image_rotary_emb: Optional[Tensor] = None,
        return_dict: bool = True,
    ):
        """
        The [`HunyuanDiT2DModel`] forward method.

        Args:
        hidden_states (`Tensor` of shape `(batch size, height, width, dim)`):
            The input tensor.
        timestep ( `Tensor`, *optional*):
            Used to indicate denoising step.
        encoder_hidden_states ( `Tensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
            Conditional embeddings for cross attention layer. This is the output of `BertModel`.
        text_embedding_mask: Tensor
            An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. This is the output
            of `BertModel`.
        encoder_hidden_states_t5 ( `Tensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
            Conditional embeddings for cross attention layer. This is the output of T5 Text Encoder.
        text_embedding_mask_t5: Tensor
            An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. This is the output
            of T5 Text Encoder.
        image_meta_size (Tensor):
            Conditional embedding indicate the image sizes
        style: Tensor:
            Conditional embedding indicate the style
        image_rotary_emb (`Tensor`):
            The image rotary embeddings to apply on query and key tensors during attention calculation.
        return_dict: bool
            Whether to return a dictionary.
        """
        patch_size = self.pos_embed.patch_size
        height, width = (
            ops.size()(hidden_states, dim=1)._attrs["int_var"] / patch_size,
            ops.size()(hidden_states, dim=2)._attrs["int_var"] / patch_size,
        )

        hidden_states = self.pos_embed(hidden_states)

        temb = self.time_extra_emb(
            timestep,
            encoder_hidden_states_t5,
            image_meta_size,
            style,
            hidden_dtype=timestep.dtype(),
        )  # [B, D]

        # text projection
        batch_size, sequence_length, _ = ops.size()(encoder_hidden_states_t5)
        encoder_hidden_states_t5 = self.text_embedder(
            ops.reshape()(
                encoder_hidden_states_t5,
                [-1, ops.size()(encoder_hidden_states_t5, dim=-1)],
            )
        )
        encoder_hidden_states_t5 = ops.reshape()(
            encoder_hidden_states_t5, [batch_size, sequence_length, -1]
        )

        encoder_hidden_states = ops.concatenate()(
            [encoder_hidden_states, encoder_hidden_states_t5], dim=1
        )
        text_embedding_mask = ops.concatenate()(
            [text_embedding_mask, text_embedding_mask_t5], dim=-1
        )
        text_embedding_mask = ops.cast()(
            ops.unsqueeze(2)(text_embedding_mask), dtype="bool"
        )

        encoder_hidden_states = ops.where()(
            text_embedding_mask,
            encoder_hidden_states,
            self.text_embedding_padding.tensor(),
        )

        skips = []
        for layer, block in enumerate(self.blocks):
            if layer > self.config.num_layers // 2:
                skip = skips.pop()
                hidden_states = block(
                    hidden_states,
                    temb=temb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_rotary_emb=image_rotary_emb,
                    skip=skip,
                )  # (N, L, D)
            else:
                hidden_states = block(
                    hidden_states,
                    temb=temb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_rotary_emb=image_rotary_emb,
                )  # (N, L, D)

            if layer < (self.config.num_layers // 2 - 1):
                skips.append(hidden_states)

        # final layer
        hidden_states = self.norm_out(hidden_states, ops.cast()(temb, dtype="float32"))
        hidden_states = self.proj_out(hidden_states)
        # (N, L, patch_size ** 2 * out_channels)

        # unpatchify: (N, out_channels, H, W)

        hidden_states = ops.reshape()(
            hidden_states,
            [
                -1,
                height,
                width,
                patch_size,
                patch_size,
                self.out_channels,
            ],
        )
        hidden_states = ops.permute()(
            hidden_states, [0, 1, 3, 2, 4, 5]
        )  # torch: nhwpqc->nchpwq
        output = ops.reshape()(
            hidden_states,
            [-1, height * self.patch_size, width * self.patch_size, self.out_channels],
        )
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
