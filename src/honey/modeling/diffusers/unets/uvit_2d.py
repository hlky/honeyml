from typing import Dict, Union

from honey.compiler import ops

from honey.frontend import nn, Tensor

from ..attention import BasicTransformerBlock, SkipFFTransformerBlock
from ..attention_processor import AttentionProcessor
from ..embeddings import get_timestep_embedding, TimestepEmbedding

from ..normalization import GlobalResponseNorm, RMSNorm
from ..resnet import Downsample2D, Upsample2D


class UVit2DModel(nn.Module):
    _supports_gradient_checkpointing = True

    def __init__(
        self,
        # global config
        hidden_size: int = 1024,
        use_bias: bool = False,
        hidden_dropout: float = 0.0,
        # conditioning dimensions
        cond_embed_dim: int = 768,
        micro_cond_encode_dim: int = 256,
        micro_cond_embed_dim: int = 1280,
        encoder_hidden_size: int = 768,
        # num tokens
        vocab_size: int = 8256,  # codebook_size + 1 (for the mask token) rounded
        codebook_size: int = 8192,
        # `UVit2DConvEmbed`
        in_channels: int = 768,
        block_out_channels: int = 768,
        num_res_blocks: int = 3,
        downsample: bool = False,
        upsample: bool = False,
        block_num_heads: int = 12,
        # `TransformerLayer`
        num_hidden_layers: int = 22,
        num_attention_heads: int = 16,
        # `Attention`
        attention_dropout: float = 0.0,
        # `FeedForward`
        intermediate_size: int = 2816,
        # `Norm`
        layer_norm_eps: float = 1e-6,
        ln_elementwise_affine: bool = True,
        sample_size: int = 64,
        dtype: str = "float16",
        **kwargs,
    ):
        super().__init__()
        self.micro_cond_encode_dim = micro_cond_encode_dim

        self.encoder_proj = nn.Linear(
            encoder_hidden_size, hidden_size, bias=use_bias, dtype=dtype
        )
        self.encoder_proj_layer_norm = RMSNorm(
            hidden_size, layer_norm_eps, ln_elementwise_affine, dtype=dtype
        )

        self.embed = UVit2DConvEmbed(
            in_channels,
            block_out_channels,
            vocab_size,
            ln_elementwise_affine,
            layer_norm_eps,
            use_bias,
            dtype=dtype,
        )

        self.cond_embed = TimestepEmbedding(
            micro_cond_embed_dim + cond_embed_dim,
            hidden_size,
            sample_proj_bias=use_bias,
            dtype=dtype,
        )

        self.down_block = UVitBlock(
            block_out_channels,
            num_res_blocks,
            hidden_size,
            hidden_dropout,
            ln_elementwise_affine,
            layer_norm_eps,
            use_bias,
            block_num_heads,
            attention_dropout,
            downsample,
            False,
            dtype=dtype,
        )

        self.project_to_hidden_norm = RMSNorm(
            block_out_channels, layer_norm_eps, ln_elementwise_affine, dtype=dtype
        )
        self.project_to_hidden = nn.Linear(
            block_out_channels, hidden_size, bias=use_bias, dtype=dtype
        )

        self.transformer_layers = nn.ModuleList(
            [
                BasicTransformerBlock(
                    dim=hidden_size,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=hidden_size // num_attention_heads,
                    dropout=hidden_dropout,
                    cross_attention_dim=hidden_size,
                    attention_bias=use_bias,
                    norm_type="ada_norm_continuous",
                    ada_norm_continous_conditioning_embedding_dim=hidden_size,
                    norm_elementwise_affine=ln_elementwise_affine,
                    norm_eps=layer_norm_eps,
                    ada_norm_bias=use_bias,
                    ff_inner_dim=intermediate_size,
                    ff_bias=use_bias,
                    attention_out_bias=use_bias,
                    dtype=dtype,
                )
                for _ in range(num_hidden_layers)
            ]
        )

        self.project_from_hidden_norm = RMSNorm(
            hidden_size, layer_norm_eps, ln_elementwise_affine, dtype=dtype
        )
        self.project_from_hidden = nn.Linear(
            hidden_size, block_out_channels, bias=use_bias, dtype=dtype
        )

        self.up_block = UVitBlock(
            block_out_channels,
            num_res_blocks,
            hidden_size,
            hidden_dropout,
            ln_elementwise_affine,
            layer_norm_eps,
            use_bias,
            block_num_heads,
            attention_dropout,
            downsample=False,
            upsample=upsample,
            dtype=dtype,
        )

        self.mlm_layer = ConvMlmLayer(
            block_out_channels,
            in_channels,
            use_bias,
            ln_elementwise_affine,
            layer_norm_eps,
            codebook_size,
            dtype=dtype,
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: Tensor,
        encoder_hidden_states: Tensor,
        pooled_text_emb: Tensor,
        micro_conds: Tensor,
        cross_attention_kwargs=None,
    ):
        encoder_hidden_states = self.encoder_proj(encoder_hidden_states)
        encoder_hidden_states = self.encoder_proj_layer_norm(encoder_hidden_states)

        micro_cond_embeds = get_timestep_embedding(
            ops.flatten()(micro_conds),
            self.micro_cond_encode_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )

        micro_cond_embeds = ops.reshape()(
            micro_cond_embeds, [ops.size()(input_ids, dim=0), -1]
        )

        pooled_text_emb = ops.concatenate()([pooled_text_emb, micro_cond_embeds], dim=1)
        pooled_text_emb = ops.cast()(pooled_text_emb, dtype=self.dtype())
        pooled_text_emb = ops.cast()(
            self.cond_embed(pooled_text_emb), encoder_hidden_states.dtype()
        )

        hidden_states = self.embed(input_ids)

        hidden_states = self.down_block(
            hidden_states,
            pooled_text_emb=pooled_text_emb,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs=cross_attention_kwargs,
        )

        batch_size, height, width, channels = ops.size()(hidden_states)
        hidden_states = ops.reshape()(
            hidden_states, [batch_size, height * width, channels]
        )

        hidden_states = self.project_to_hidden_norm(hidden_states)
        hidden_states = self.project_to_hidden(hidden_states)

        for layer in self.transformer_layers:
            hidden_states = layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs={"pooled_text_emb": pooled_text_emb},
            )

        hidden_states = self.project_from_hidden_norm(hidden_states)
        hidden_states = self.project_from_hidden(hidden_states)

        hidden_states = ops.reshape()(
            hidden_states, [batch_size, height, width, channels]
        )

        hidden_states = self.up_block(
            hidden_states,
            pooled_text_emb=pooled_text_emb,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs=cross_attention_kwargs,
        )

        logits = self.mlm_layer(hidden_states)

        return logits

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


class UVit2DConvEmbed(nn.Module):
    def __init__(
        self,
        in_channels: int,
        block_out_channels: int,
        vocab_size: int,
        elementwise_affine: bool,
        eps: float,
        bias: bool,
        dtype: str = "float16",
    ):
        super().__init__()
        self.embeddings = nn.Embedding([vocab_size, in_channels], dtype=dtype)
        self.layer_norm = RMSNorm(in_channels, eps, elementwise_affine, dtype=dtype)
        self.conv = nn.Conv2d(
            in_channels, block_out_channels, kernel_size=1, dtype=dtype, bias=bias
        )

    def forward(self, input_ids):
        embeddings = self.embeddings(input_ids)
        embeddings = self.layer_norm(embeddings)  # NOTE: torch .permute(0, 3, 1, 2)
        embeddings = self.conv(embeddings)
        return embeddings


class UVitBlock(nn.Module):
    def __init__(
        self,
        channels,
        num_res_blocks: int,
        hidden_size,
        hidden_dropout,
        ln_elementwise_affine,
        layer_norm_eps,
        use_bias,
        block_num_heads,
        attention_dropout,
        downsample: bool,
        upsample: bool,
        dtype: str = "float16",
    ):
        super().__init__()

        if downsample:
            self.downsample = Downsample2D(
                channels,
                use_conv=True,
                padding=0,
                name="Conv2d_0",
                kernel_size=2,
                norm_type="rms_norm",
                eps=layer_norm_eps,
                elementwise_affine=ln_elementwise_affine,
                bias=use_bias,
                dtype=dtype,
            )
        else:
            self.downsample = None

        self.res_blocks = nn.ModuleList(
            [
                ConvNextBlock(
                    channels,
                    layer_norm_eps,
                    ln_elementwise_affine,
                    use_bias,
                    hidden_dropout,
                    hidden_size,
                    dtype=dtype,
                )
                for i in range(num_res_blocks)
            ]
        )

        self.attention_blocks = nn.ModuleList(
            [
                SkipFFTransformerBlock(
                    channels,
                    block_num_heads,
                    channels // block_num_heads,
                    hidden_size,
                    use_bias,
                    attention_dropout,
                    channels,
                    attention_bias=use_bias,
                    attention_out_bias=use_bias,
                    dtype=dtype,
                )
                for _ in range(num_res_blocks)
            ]
        )

        if upsample:
            self.upsample = Upsample2D(
                channels,
                use_conv_transpose=True,
                kernel_size=2,
                padding=0,
                name="conv",
                norm_type="rms_norm",
                eps=layer_norm_eps,
                elementwise_affine=ln_elementwise_affine,
                bias=use_bias,
                interpolate=False,
                dtype=dtype,
            )
        else:
            self.upsample = None

    def forward(
        self,
        x: Tensor,
        pooled_text_emb: Tensor,
        encoder_hidden_states: Tensor,
        cross_attention_kwargs,
    ):
        if self.downsample is not None:
            x = self.downsample(x)

        for res_block, attention_block in zip(self.res_blocks, self.attention_blocks):
            x = res_block(x, pooled_text_emb)

            batch_size, height, width, channels = ops.size()(x)
            x = ops.reshape()(x, [batch_size, height * width, channels])
            x = attention_block(
                x,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
            )
            x = ops.reshape()(x, [batch_size, height, width, channels])

        if self.upsample is not None:
            x = self.upsample(x)

        return x


class ConvNextBlock(nn.Module):
    def __init__(
        self,
        channels,
        layer_norm_eps,
        ln_elementwise_affine,
        use_bias,
        hidden_dropout,
        hidden_size,
        res_ffn_factor=4,
        dtype: str = "float16",
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            dtype=dtype,
            bias=use_bias,
        )
        self.norm = RMSNorm(
            channels, layer_norm_eps, ln_elementwise_affine, dtype=dtype
        )
        self.channelwise_linear_1 = nn.Linear(
            channels, int(channels * res_ffn_factor), bias=use_bias, dtype=dtype
        )
        self.channelwise_act = ops.gelu
        self.channelwise_norm = GlobalResponseNorm(
            int(channels * res_ffn_factor), dtype=dtype
        )
        self.channelwise_linear_2 = nn.Linear(
            int(channels * res_ffn_factor), channels, bias=use_bias, dtype=dtype
        )
        self.channelwise_dropout = nn.Dropout(hidden_dropout)
        self.cond_embeds_mapper = nn.Linear(
            hidden_size, channels * 2, use_bias, dtype=dtype
        )

    def forward(self, x: Tensor, cond_embeds: Tensor):
        x_res = x

        x = self.depthwise(x)

        x = self.norm(x)

        x = self.channelwise_linear_1(x)
        x = self.channelwise_act(x)
        x = self.channelwise_norm(x)
        x = self.channelwise_linear_2(x)
        x = self.channelwise_dropout(x)

        x = x + x_res

        scale, shift = ops.chunk()(
            self.cond_embeds_mapper(ops.silu(cond_embeds)), 2, dim=1
        )
        x = x * (1 + ops.unsqueeze(-1)(ops.unsqueeze(-1)(scale))) + ops.unsqueeze(-1)(
            ops.unsqueeze(-1)(shift)
        )

        return x


class ConvMlmLayer(nn.Module):
    def __init__(
        self,
        block_out_channels: int,
        in_channels: int,
        use_bias: bool,
        ln_elementwise_affine: bool,
        layer_norm_eps: float,
        codebook_size: int,
        dtype: str = "float16",
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            block_out_channels, in_channels, kernel_size=1, dtype=dtype, bias=use_bias
        )
        self.layer_norm = RMSNorm(in_channels, layer_norm_eps, ln_elementwise_affine)
        self.conv2 = nn.Conv2d(
            in_channels, codebook_size, kernel_size=1, dtype=dtype, bias=use_bias
        )

    def forward(self, hidden_states):
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.layer_norm(hidden_states)  # NOTE: pytorch permute
        logits = self.conv2(hidden_states)
        return logits
