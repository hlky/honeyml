from typing import Any, Dict, List, Optional, Union

from honey.compiler import ops

from honey.frontend import IntVar, nn, Tensor

from ..attention import Attention, FeedForward
from ..attention_processor import FluxAttnProcessor2_0, FluxSingleAttnProcessor2_0
from ..embeddings import (
    CombinedTimestepGuidanceTextProjEmbeddings,
    CombinedTimestepTextProjEmbeddings,
    GELU,
)
from ..normalization import (
    AdaLayerNormContinuous,
    AdaLayerNormZero,
    AdaLayerNormZeroSingle,
)
from .transformer_2d import Transformer2DModelOutput

from ..modeling_outputs import Transformer2DModelOutput


def get_shape(x):
    shape = [
        (
            it.value()
            if not isinstance(it, IntVar)
            else [it.lower_bound(), it.upper_bound()]
        )
        for it in x._attrs["shape"]
    ]
    return shape


# YiYi to-do: refactor rope related functions/classes
def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0, "The dimension must be even."

    scale = ops.cast()(ops.arange(0, dim, 2)(), "float32") / dim
    omega = 1.0 / ops.pow(theta, scale)

    batch_size = ops.size()(pos, dim=0)
    pos = ops.reshape()(pos, [-1, 1])  # (M, 1)
    omega = ops.reshape()(omega, [1, -1])  # (1, D/2)
    out = ops.gemm_rrr()(ops.cast()(pos, "float32"), omega)
    cos_out = ops.cos(out)
    sin_out = ops.sin(out)

    stacked_out = ops.stack()([cos_out, -sin_out, sin_out, cos_out], dim=-1)
    out = ops.reshape()(stacked_out, [batch_size, -1, dim // 2, 2, 2])
    return ops.cast()(out, "float32")


# YiYi to-do: refactor rope related functions/classes
class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: List[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_dim = len(ops.size()(ids))
        n_axes = ops.size()(ids, dim=-1)._attrs["int_var"].symbolic_value()
        emb = ops.concatenate()(
            [
                rope(
                    ops.dynamic_slice()(
                        ids,
                        start_indices=[0] * (n_dim - 1) + [i],
                        end_indices=[None] * (n_dim - 1) + [i + 1],
                    ),
                    self.axes_dim[i],
                    self.theta,
                )
                for i in range(n_axes)
            ],
            dim=-3,
        )

        return ops.unsqueeze(1)(emb)


class FluxSingleTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.
    Reference: https://arxiv.org/abs/2403.03206
    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        mlp_ratio=4.0,
        dtype: str = "float16",
    ):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim, dtype=dtype)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim, dtype=dtype)
        self.act_mlp = GELU()
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim, dtype=dtype)

        processor = FluxSingleAttnProcessor2_0()
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=processor,
            qk_norm="rms_norm",
            eps=1e-6,
            pre_only=True,
            dtype=dtype,
        )

    def forward(
        self,
        hidden_states: Tensor,
        temb: Tensor,
        image_rotary_emb=None,
    ):
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = ops.concatenate()([attn_output, mlp_hidden_states], dim=2)
        hidden_states = ops.unsqueeze(1)(gate) * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class FluxTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.
    Reference: https://arxiv.org/abs/2403.03206
    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        qk_norm="rms_norm",
        eps=1e-6,
        dtype: str = "float16",
    ):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim, dtype=dtype)

        self.norm1_context = AdaLayerNormZero(dim, dtype=dtype)

        processor = FluxAttnProcessor2_0()
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=processor,
            qk_norm=qk_norm,
            eps=eps,
            dtype=dtype,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6, dtype=dtype)
        self.ff = FeedForward(
            dim=dim, dim_out=dim, activation_fn="gelu-approximate", dtype=dtype
        )

        self.norm2_context = nn.LayerNorm(
            dim, elementwise_affine=False, eps=1e-6, dtype=dtype
        )
        self.ff_context = FeedForward(
            dim=dim, dim_out=dim, activation_fn="gelu-approximate", dtype=dtype
        )

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        temb: Tensor,
        image_rotary_emb=None,
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
            self.norm1_context(encoder_hidden_states, emb=temb)
        )

        # Attention.
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = ops.unsqueeze(1)(gate_msa) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (
            1 + ops.unsqueeze(1)(scale_mlp)
        ) + ops.unsqueeze(1)(shift_mlp)

        ff_output = self.ff(norm_hidden_states)
        ff_output = ops.unsqueeze(1)(gate_mlp) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.

        context_attn_output = ops.unsqueeze(1)(c_gate_msa) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (
            1 + ops.unsqueeze(1)(c_scale_mlp)
        ) + ops.unsqueeze(1)(c_shift_mlp)

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = (
            encoder_hidden_states + ops.unsqueeze(1)(c_gate_mlp) * context_ff_output
        )

        return encoder_hidden_states, hidden_states


class FluxTransformer2DModel(nn.Module):
    """
    The Transformer model introduced in Flux.
    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/
    Parameters:
        patch_size (`int`): Patch size to turn the input data into small patches.
        in_channels (`int`, *optional*, defaults to 16): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 18): The number of layers of MMDiT blocks to use.
        num_single_layers (`int`, *optional*, defaults to 18): The number of layers of single DiT blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 18): The number of heads to use for multi-head attention.
        joint_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        pooled_projection_dim (`int`): Number of dimensions to use when projecting the `pooled_projections`.
        guidance_embeds (`bool`, defaults to False): Whether to use guidance embeddings.
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        dtype: str = "float16",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_layers = num_layers
        self.num_single_layers = num_single_layers
        self.attention_head_dim = attention_head_dim
        self.num_attention_heads = num_attention_heads
        self.joint_attention_dim = joint_attention_dim
        self.pooled_projection_dim = pooled_projection_dim
        self.inner_dim = self.num_attention_heads * self.attention_head_dim

        self.pos_embed = EmbedND(dim=self.inner_dim, theta=10000, axes_dim=[16, 56, 56])
        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings
            if guidance_embeds
            else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim,
            pooled_projection_dim=self.pooled_projection_dim,
            dtype=dtype,
        )

        self.context_embedder = nn.Linear(
            self.joint_attention_dim, self.inner_dim, dtype=dtype
        )
        self.x_embedder = nn.Linear(self.in_channels, self.inner_dim, dtype=dtype)

        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.num_attention_heads,
                    attention_head_dim=self.attention_head_dim,
                    dtype=dtype,
                )
                for i in range(self.num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.num_attention_heads,
                    attention_head_dim=self.attention_head_dim,
                    dtype=dtype,
                )
                for i in range(self.num_single_layers)
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

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor = None,
        pooled_projections: Tensor = None,
        timestep: Tensor = None,
        img_ids: Tensor = None,
        txt_ids: Tensor = None,
        guidance: Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[Tensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.
        Args:
            hidden_states (`Tensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`Tensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`Tensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.
        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """

        hidden_states = self.x_embedder(hidden_states)

        timestep = ops.cast()(timestep, hidden_states.dtype()) * 1000
        if guidance is not None:
            guidance = ops.cast()(guidance, hidden_states.dtype()) * 1000
        else:
            guidance = None
        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        ids = ops.concatenate()((txt_ids, img_ids), dim=1)
        image_rotary_emb = self.pos_embed(ids)

        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

        hidden_states = ops.concatenate()([encoder_hidden_states, hidden_states], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )
        encoder_hidden_states_dim = ops.size()(encoder_hidden_states, dim=1)._attrs[
            "int_var"
        ]
        hidden_states_dim = ops.size()(hidden_states, dim=1)._attrs["int_var"]
        hidden_states_indices = ops.cast()(
            ops.arange(encoder_hidden_states_dim, hidden_states_dim, 1)(), "int64"
        )
        hidden_states = ops.index_select(dim=1)(hidden_states, hidden_states_indices)

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
