from dataclasses import dataclass
from typing import Dict, Optional, Union

from honey.compiler import ops

from honey.frontend import nn, Tensor

from ..attention import BasicTransformerBlock
from ..attention_processor import AttentionProcessor
from ..embeddings import TimestepEmbedding, Timesteps

from ..utils import BaseOutput


@dataclass
class PriorTransformerOutput(BaseOutput):
    """
    The output of [`PriorTransformer`].

    Args:
        predicted_image_embedding (`Tensor` of shape `(batch_size, embedding_dim)`):
            The predicted CLIP image embedding conditioned on the CLIP text embedding input.
    """

    predicted_image_embedding: Tensor


class PriorTransformer(nn.Module):
    """
    A Prior Transformer model.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 32): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_layers (`int`, *optional*, defaults to 20): The number of layers of Transformer blocks to use.
        embedding_dim (`int`, *optional*, defaults to 768): The dimension of the model input `hidden_states`
        num_embeddings (`int`, *optional*, defaults to 77):
            The number of embeddings of the model input `hidden_states`
        additional_embeddings (`int`, *optional*, defaults to 4): The number of additional tokens appended to the
            projected `hidden_states`. The actual length of the used `hidden_states` is `num_embeddings +
            additional_embeddings`.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        time_embed_act_fn (`str`, *optional*, defaults to 'silu'):
            The activation function to use to create timestep embeddings.
        norm_in_type (`str`, *optional*, defaults to None): The normalization layer to apply on hidden states before
            passing to Transformer blocks. Set it to `None` if normalization is not needed.
        embedding_proj_norm_type (`str`, *optional*, defaults to None):
            The normalization layer to apply on the input `proj_embedding`. Set it to `None` if normalization is not
            needed.
        encoder_hid_proj_type (`str`, *optional*, defaults to `linear`):
            The projection layer to apply on the input `encoder_hidden_states`. Set it to `None` if
            `encoder_hidden_states` is `None`.
        added_emb_type (`str`, *optional*, defaults to `prd`): Additional embeddings to condition the model.
            Choose from `prd` or `None`. if choose `prd`, it will prepend a token indicating the (quantized) dot
            product between the text embedding and image embedding as proposed in the unclip paper
            https://arxiv.org/abs/2204.06125 If it is `None`, no additional embeddings will be prepended.
        time_embed_dim (`int, *optional*, defaults to None): The dimension of timestep embeddings.
            If None, will be set to `num_attention_heads * attention_head_dim`
        embedding_proj_dim (`int`, *optional*, default to None):
            The dimension of `proj_embedding`. If None, will be set to `embedding_dim`.
        clip_embed_dim (`int`, *optional*, default to None):
            The dimension of the output. If None, will be set to `embedding_dim`.
    """

    def __init__(
        self,
        num_attention_heads: int = 32,
        attention_head_dim: int = 64,
        num_layers: int = 20,
        embedding_dim: int = 768,
        num_embeddings=77,
        additional_embeddings=4,
        dropout: float = 0.0,
        time_embed_act_fn: str = "silu",
        norm_in_type: Optional[str] = None,  # layer
        embedding_proj_norm_type: Optional[str] = None,  # layer
        encoder_hid_proj_type: Optional[str] = "linear",  # linear
        added_emb_type: Optional[str] = "prd",  # prd
        time_embed_dim: Optional[int] = None,
        embedding_proj_dim: Optional[int] = None,
        clip_embed_dim: Optional[int] = None,
        dtype: str = "float16",
        **kwargs,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.additional_embeddings = additional_embeddings

        time_embed_dim = time_embed_dim or inner_dim
        embedding_proj_dim = embedding_proj_dim or embedding_dim
        clip_embed_dim = clip_embed_dim or embedding_dim

        self.time_proj = Timesteps(inner_dim, True, 0, dtype=dtype)
        self.time_embedding = TimestepEmbedding(
            inner_dim,
            time_embed_dim,
            out_dim=inner_dim,
            act_fn=time_embed_act_fn,
            dtype=dtype,
        )

        self.proj_in = nn.Linear(embedding_dim, inner_dim, dtype=dtype)

        if embedding_proj_norm_type is None:
            self.embedding_proj_norm = None
        elif embedding_proj_norm_type == "layer":
            self.embedding_proj_norm = nn.LayerNorm(embedding_proj_dim, dtype=dtype)
        else:
            raise ValueError(
                f"unsupported embedding_proj_norm_type: {embedding_proj_norm_type}"
            )

        self.embedding_proj = nn.Linear(embedding_proj_dim, inner_dim, dtype=dtype)

        if encoder_hid_proj_type is None:
            self.encoder_hidden_states_proj = None
        elif encoder_hid_proj_type == "linear":
            self.encoder_hidden_states_proj = nn.Linear(
                embedding_dim, inner_dim, dtype=dtype
            )
        else:
            raise ValueError(
                f"unsupported encoder_hid_proj_type: {encoder_hid_proj_type}"
            )

        self.positional_embedding = nn.Parameter(
            [1, num_embeddings + additional_embeddings, inner_dim], dtype=dtype
        )

        if added_emb_type == "prd":
            self.prd_embedding = nn.Parameter([1, 1, inner_dim], dtype=dtype)
        elif added_emb_type is None:
            self.prd_embedding = None
        else:
            raise ValueError(
                f"`added_emb_type`: {added_emb_type} is not supported. Make sure to choose one of `'prd'` or `None`."
            )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    activation_fn="gelu",
                    attention_bias=True,
                    dtype=dtype,
                )
                for d in range(num_layers)
            ]
        )

        if norm_in_type == "layer":
            self.norm_in = nn.LayerNorm(inner_dim, dtype=dtype)
        elif norm_in_type is None:
            self.norm_in = None
        else:
            raise ValueError(f"Unsupported norm_in_type: {norm_in_type}.")

        self.norm_out = nn.LayerNorm(inner_dim, dtype=dtype)

        self.proj_to_clip_embeddings = nn.Linear(inner_dim, clip_embed_dim, dtype=dtype)

        raise NotImplementedError("triu")
        # causal_attention_mask = torch.full(
        #     [
        #         num_embeddings + additional_embeddings,
        #         num_embeddings + additional_embeddings,
        #     ],
        #     -10000.0,
        # )
        # causal_attention_mask.triu_(1)
        # causal_attention_mask = causal_attention_mask[None, ...]
        self.causal_attention_mask = Tensor(
            [
                1,
                num_embeddings + additional_embeddings,
                num_embeddings + additional_embeddings,
            ],
            name="causal_attention_mask",
            dtype=dtype,
        )

        self.clip_mean = nn.Parameter([1, clip_embed_dim], dtype=dtype)
        self.clip_std = nn.Parameter([1, clip_embed_dim], dtype=dtype)

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
        proj_embedding: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        return_dict: bool = True,
    ):
        """
        The [`PriorTransformer`] forward method.

        Args:
            hidden_states (`Tensor` of shape `(batch_size, embedding_dim)`):
                The currently predicted image embeddings.
            timestep (`Tensor`):
                Current denoising step.
            proj_embedding (`Tensor` of shape `(batch_size, embedding_dim)`):
                Projected embedding vector the denoising process is conditioned on.
            encoder_hidden_states (`Tensor` of shape `(batch_size, num_embeddings, embedding_dim)`):
                Hidden states of the text embeddings the denoising process is conditioned on.
            attention_mask (`torch.BoolTensor` of shape `(batch_size, num_embeddings)`):
                Text mask for the text embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformers.prior_transformer.PriorTransformerOutput`] instead of
                a plain tuple.

        Returns:
            [`~models.transformers.prior_transformer.PriorTransformerOutput`] or `tuple`:
                If return_dict is True, a [`~models.transformers.prior_transformer.PriorTransformerOutput`] is
                returned, otherwise a tuple is returned where the first element is the sample tensor.
        """
        batch_size = ops.size()(hidden_states, dim=0)

        timesteps = timestep

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * ops.full()(
            [batch_size], fill_value=1.0, dtype=timesteps.dtype()
        )

        timesteps_projected = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might be fp16, so we need to cast here.
        timesteps_projected = ops.cast()(timesteps_projected, dtype=self.dtype)
        time_embeddings = self.time_embedding(timesteps_projected)

        if self.embedding_proj_norm is not None:
            proj_embedding = self.embedding_proj_norm(proj_embedding)

        proj_embeddings = self.embedding_proj(proj_embedding)
        if (
            self.encoder_hidden_states_proj is not None
            and encoder_hidden_states is not None
        ):
            encoder_hidden_states = self.encoder_hidden_states_proj(
                encoder_hidden_states
            )
        elif (
            self.encoder_hidden_states_proj is not None
            and encoder_hidden_states is None
        ):
            raise ValueError(
                "`encoder_hidden_states_proj` requires `encoder_hidden_states` to be set"
            )

        hidden_states = self.proj_in(hidden_states)

        positional_embeddings = ops.cast()(
            self.positional_embedding.tensor(), dtype=hidden_states.dtype()
        )

        additional_embeds = []
        additional_embeddings_len = 0

        if encoder_hidden_states is not None:
            additional_embeds.append(encoder_hidden_states)
            additional_embeddings_len += ops.size()(encoder_hidden_states, dim=1)

        if len(ops.size()(proj_embeddings)) == 2:
            proj_embeddings = ops.unsqueeze(1)(proj_embeddings)

        if len(ops.size()(hidden_states)) == 2:
            hidden_states = ops.unsqueeze(1)(hidden_states)

        additional_embeds = additional_embeds + [
            proj_embeddings,
            ops.unsqueeze(1)(time_embeddings),
            hidden_states,
        ]

        if self.prd_embedding is not None:
            prd_embedding = ops.expand()(
                ops.cast()(self.prd_embedding.tensor(), dtype=hidden_states.dtype()),
                [batch_size, -1, -1],
            )
            additional_embeds.append(prd_embedding)

        hidden_states = ops.concatenate()(
            additional_embeds,
            dim=1,
        )

        # Allow positional_embedding to not include the `addtional_embeddings` and instead pad it with zeros for these additional tokens
        additional_embeddings_len = (
            additional_embeddings_len + ops.size()(proj_embeddings, dim=1) + 1
        )
        if ops.size()(positional_embeddings, dim=1) < ops.size()(hidden_states, dim=1):
            positional_embeddings = ops.pad(
                (
                    0,
                    0,
                    additional_embeddings_len.upper_bound(),  # NOTE: verify
                    1 if self.prd_embedding is not None else 0,
                ),
                value=0.0,
            )(positional_embeddings)

        hidden_states = hidden_states + positional_embeddings

        if attention_mask is not None:
            attention_mask = (
                1 - ops.cast()(attention_mask, hidden_states.dtype())
            ) * -10000.0
            attention_mask = ops.pad(
                (0, self.additional_embeddings), mode="constant", value=0.0
            )(attention_mask)
            attention_mask = ops.cast()(
                ops.unsqueeze(1)(attention_mask) + self.causal_attention_mask,
                hidden_states.dtype(),
            )
            attention_mask = ops.repeat_interleave(
                repeats=self.num_attention_heads, dim=0
            )(attention_mask)

        if self.norm_in is not None:
            hidden_states = self.norm_in(hidden_states)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask)

        hidden_states = self.norm_out(hidden_states)

        if self.prd_embedding is not None:
            hidden_states = ops.dynamic_slice()(
                hidden_states, start_indices=[0, -1], end_indices=[None, None]
            )
        else:
            hidden_states = ops.dynamic_slice()(
                hidden_states,
                start_indices=[0, additional_embeddings_len],
                end_indices=[None, None],
            )

        predicted_image_embedding = self.proj_to_clip_embeddings(hidden_states)

        if not return_dict:
            return (predicted_image_embedding,)

        return PriorTransformerOutput(
            predicted_image_embedding=predicted_image_embedding
        )

    def post_process_latents(self, prior_latents):
        prior_latents = (prior_latents * self.clip_std) + self.clip_mean
        return prior_latents
