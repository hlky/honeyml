from typing import Any, Dict, Optional

from honey.compiler import ops

from honey.frontend import IntVar, nn, Tensor

from .activations import ApproximateGELU, GEGLU, GELU
from .attention_processor import Attention, JointAttnProcessor2_0
from .embeddings import SinusoidalPositionalEmbedding
from .normalization import (
    AdaLayerNorm,
    AdaLayerNormContinuous,
    AdaLayerNormZero,
    RMSNorm,
)


def get_shape(x):
    shape = [
        it.value() if not isinstance(it, IntVar) else it.symbolic_value()
        for it in x._attrs["shape"]
    ]
    return shape


class GatedSelfAttentionDense(nn.Module):
    r"""
    A gated self-attention dense layer that combines visual features and object features.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        context_dim (`int`): The number of channels in the context.
        n_heads (`int`): The number of heads to use for attention.
        d_head (`int`): The number of channels in each head.
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        n_heads: int,
        d_head: int,
        dtype: str = "float16",
    ):
        super().__init__()

        # we need a linear projection since we need cat visual feature and obj feature
        self.linear = nn.Linear(context_dim, query_dim, dtype=dtype)

        self.attn = Attention(
            query_dim=query_dim, heads=n_heads, dim_head=d_head, dtype=dtype
        )
        self.ff = FeedForward(query_dim, activation_fn="geglu", dtype=dtype)

        self.norm1 = nn.LayerNorm(query_dim, dtype=dtype)
        self.norm2 = nn.LayerNorm(query_dim, dtype=dtype)

        self.alpha_attn = nn.Parameter([1], value=0.0, dtype=dtype)
        self.alpha_dense = nn.Parameter([1], value=0.0, dtype=dtype)

        self.enabled = True

    def forward(self, x: Tensor, objs: Tensor) -> Tensor:
        if not self.enabled:
            return x

        n_visual = ops.size()(x, dim=1)
        objs = self.linear(objs)

        x = x + ops.tanh(self.alpha_attn.tensor()) * ops.dynamic_slice()(
            self.attn(self.norm1(ops.concatenate()([x, objs], dim=1))),
            start_indices=[0, 0, 0],
            end_indices=[None, n_visual, None],
        )
        x = x + ops.tanh(self.alpha_dense.tensor()) * self.ff(self.norm2(x))

        return x


class JointTransformerBlock(nn.Module):
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
        context_pre_only=False,
        dtype: str = "float16",
    ):
        super().__init__()

        self.context_pre_only = context_pre_only
        context_norm_type = (
            "ada_norm_continous" if context_pre_only else "ada_norm_zero"
        )

        self.norm1 = AdaLayerNormZero(dim, dtype=dtype)

        if context_norm_type == "ada_norm_continous":
            self.norm1_context = AdaLayerNormContinuous(
                dim,
                dim,
                elementwise_affine=False,
                eps=1e-6,
                bias=True,
                norm_type="layer_norm",
                dtype=dtype,
            )
        elif context_norm_type == "ada_norm_zero":
            self.norm1_context = AdaLayerNormZero(dim, dtype=dtype)
        else:
            raise ValueError(
                f"Unknown context_norm_type: {context_norm_type}, currently only support `ada_norm_continous`, `ada_norm_zero`"
            )
        processor = JointAttnProcessor2_0()

        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim // num_attention_heads,
            heads=num_attention_heads,
            out_dim=attention_head_dim,
            context_pre_only=context_pre_only,
            bias=True,
            processor=processor,
            dtype=dtype,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6, dtype=dtype)
        self.ff = FeedForward(
            dim=dim, dim_out=dim, activation_fn="gelu-approximate", dtype=dtype
        )

        if not context_pre_only:
            self.norm2_context = nn.LayerNorm(
                dim, elementwise_affine=False, eps=1e-6, dtype=dtype
            )
            self.ff_context = FeedForward(
                dim=dim, dim_out=dim, activation_fn="gelu-approximate", dtype=dtype
            )
        else:
            self.norm2_context = None
            self.ff_context = None

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def forward(
        self, hidden_states: Tensor, encoder_hidden_states: Tensor, temb: Tensor
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )

        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
        else:
            (
                norm_encoder_hidden_states,
                c_gate_msa,
                c_shift_mlp,
                c_scale_mlp,
                c_gate_mlp,
            ) = self.norm1_context(encoder_hidden_states, emb=temb)

        # Attention.
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
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
        if self.context_pre_only:
            encoder_hidden_states = None
        else:
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


class BasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        upcast_attention (`bool`, *optional*):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The normalization layer to use. Can be `"layer_norm"`, `"ada_norm"` or `"ada_norm_zero"`.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        attention_type (`str`, *optional*, defaults to `"default"`):
            The type of attention to use. Can be `"default"` or `"gated"` or `"gated-text-image"`.
        positional_embeddings (`str`, *optional*, defaults to `None`):
            The type of positional embeddings to apply to.
        num_positional_embeddings (`int`, *optional*, defaults to `None`):
            The maximum number of positional embeddings to apply.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ada_norm_continous_conditioning_embedding_dim: Optional[int] = None,
        ada_norm_bias: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        dtype: str = "float16",
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention

        # We keep these boolean flags for backward-compatibility.
        self.use_ada_layer_norm_zero = (
            num_embeds_ada_norm is not None
        ) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (
            num_embeds_ada_norm is not None
        ) and norm_type == "ada_norm"
        self.use_ada_layer_norm_single = norm_type == "ada_norm_single"
        self.use_layer_norm = norm_type == "layer_norm"
        self.use_ada_layer_norm_continuous = norm_type == "ada_norm_continuous"

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        self.norm_type = norm_type
        self.num_embeds_ada_norm = num_embeds_ada_norm

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(
                dim, max_seq_length=num_positional_embeddings, dtype=dtype
            )
        else:
            self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm, dtype=dtype)
        elif norm_type == "ada_norm_zero":
            self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm, dtype=dtype)
        elif norm_type == "ada_norm_continuous":
            self.norm1 = AdaLayerNormContinuous(
                dim,
                ada_norm_continous_conditioning_embedding_dim,
                norm_elementwise_affine,
                norm_eps,
                ada_norm_bias,
                "rms_norm",
                dtype=dtype,
            )
        else:
            self.norm1 = nn.LayerNorm(
                dim,
                elementwise_affine=norm_elementwise_affine,
                eps=norm_eps,
                dtype=dtype,
            )

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
            dtype=dtype,
        )

        # 2. Cross-Attn
        if cross_attention_dim is not None or double_self_attention:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            if norm_type == "ada_norm":
                self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm, dtype=dtype)
            elif norm_type == "ada_norm_continuous":
                self.norm2 = AdaLayerNormContinuous(
                    dim,
                    ada_norm_continous_conditioning_embedding_dim,
                    norm_elementwise_affine,
                    norm_eps,
                    ada_norm_bias,
                    "rms_norm",
                    dtype=dtype,
                )
            else:
                self.norm2 = nn.LayerNorm(
                    dim,
                    norm_eps,
                    elementwise_affine=norm_elementwise_affine,
                    dtype=dtype,
                )

            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=(
                    cross_attention_dim if not double_self_attention else None
                ),
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                out_bias=attention_out_bias,
                dtype=dtype,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.norm2 = None
            self.attn2 = None

        # 3. Feed-forward
        if norm_type == "ada_norm_continuous":
            self.norm3 = AdaLayerNormContinuous(
                dim,
                ada_norm_continous_conditioning_embedding_dim,
                norm_elementwise_affine,
                norm_eps,
                ada_norm_bias,
                "layer_norm",
                dtype=dtype,
            )

        elif norm_type in [
            "ada_norm_zero",
            "ada_norm",
            "layer_norm",
            "ada_norm_continuous",
        ]:
            self.norm3 = nn.LayerNorm(
                dim, norm_eps, elementwise_affine=norm_elementwise_affine, dtype=dtype
            )
        elif norm_type == "layer_norm_i2vgen":
            self.norm3 = None

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
            dtype=dtype,
        )

        # 4. Fuser
        if attention_type == "gated" or attention_type == "gated-text-image":
            self.fuser = GatedSelfAttentionDense(
                dim, cross_attention_dim, num_attention_heads, attention_head_dim
            )

        # 5. Scale-shift for PixArt-Alpha.
        if norm_type == "ada_norm_single":
            self.scale_shift_table = nn.Parameter([6, dim], dtype=dtype)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        timestep: Optional[Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[Tensor] = None,
        added_cond_kwargs: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = ops.size()(hidden_states, dim=0)

        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.norm_type == "ada_norm_zero":
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states,
                timestep,
                class_labels,
                hidden_dtype=hidden_states.dtype(),
            )
        elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm1(
                hidden_states, added_cond_kwargs["pooled_text_emb"]
            )
        elif self.norm_type == "ada_norm_single":
            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ) = ops.chunk()(
                ops.unsqueeze(0)(self.scale_shift_table.tensor())
                + ops.reshape()(timestep, [batch_size, 6, -1]),
                chunks=6,
                dim=1,
            )
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            norm_hidden_states = ops.squeeze(1)(norm_hidden_states)
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # 1. Prepare GLIGEN inputs
        cross_attention_kwargs = (
            cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        )
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=(
                encoder_hidden_states if self.only_cross_attention else None
            ),
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        if self.norm_type == "ada_norm_zero":
            attn_output = ops.unsqueeze(1)(gate_msa) * attn_output
        elif self.norm_type == "ada_norm_single":
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if len(ops.size()(hidden_states)) == 4:
            hidden_states = ops.squeeze(1)(hidden_states)

        # 1.2 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if self.attn2 is not None:
            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm2(hidden_states)
            elif self.norm_type == "ada_norm_single":
                # For PixArt norm2 isn't applied here:
                # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                norm_hidden_states = hidden_states
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm2(
                    hidden_states, added_cond_kwargs["pooled_text_emb"]
                )
            else:
                raise ValueError("Incorrect norm")

            if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        # i2vgen doesn't have this norm 🤷‍♂️
        if self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm3(
                hidden_states, added_cond_kwargs["pooled_text_emb"]
            )
        elif not self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm3(hidden_states)

        if self.norm_type == "ada_norm_zero":
            norm_hidden_states = norm_hidden_states * (
                1 + ops.unsqueeze(1)(scale_mlp)
            ) + ops.unsqueeze(1)(shift_mlp)

        if self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states)

        if self.norm_type == "ada_norm_zero":
            ff_output = ops.unsqueeze(1)(gate_mlp) * ff_output
        elif self.norm_type == "ada_norm_single":
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if len(ops.size()(hidden_states)) == 4:
            hidden_states = ops.squeeze(1)(hidden_states)

        return hidden_states


class TemporalBasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block for video like data.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        time_mix_inner_dim (`int`): The number of channels for temporal attention.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
    """

    def __init__(
        self,
        dim: int,
        time_mix_inner_dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: Optional[int] = None,
        dtype: str = "float16",
    ):
        super().__init__()
        self.is_res = dim == time_mix_inner_dim

        self.norm_in = nn.LayerNorm(dim, dtype=dtype)

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        self.ff_in = FeedForward(
            dim, dim_out=time_mix_inner_dim, activation_fn="geglu", dtype=dtype
        )

        self.norm1 = nn.LayerNorm(time_mix_inner_dim, dtype=dtype)
        self.attn1 = Attention(
            query_dim=time_mix_inner_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            cross_attention_dim=None,
            dtype=dtype,
        )

        # 2. Cross-Attn
        if cross_attention_dim is not None:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            self.norm2 = nn.LayerNorm(time_mix_inner_dim, dtype=dtype)
            self.attn2 = Attention(
                query_dim=time_mix_inner_dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dtype=dtype,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.norm2 = None
            self.attn2 = None

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(time_mix_inner_dim, dtype=dtype)
        self.ff = FeedForward(time_mix_inner_dim, activation_fn="geglu", dtype=dtype)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = None

    def set_chunk_feed_forward(self, chunk_size: Optional[int], **kwargs):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        # chunk dim should be hardcoded to 1 to have better speed vs. memory trade-off
        self._chunk_dim = 1

    def forward(
        self,
        hidden_states: Tensor,
        num_frames: int,
        encoder_hidden_states: Optional[Tensor] = None,
    ) -> Tensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = ops.size()(hidden_states, dim=0)

        batch_frames, seq_length, channels = ops.size()(hidden_states)
        batch_size = batch_frames / num_frames

        hidden_states = ops.reshape()(
            ops.unsqueeze(0)(hidden_states),
            [batch_size, num_frames, seq_length, channels],
        )
        # hidden_states = hidden_states.permute(0, 2, 1, 3)
        hidden_states = ops.permute0213()(hidden_states)
        hidden_states = ops.reshape()(
            hidden_states, [batch_size * seq_length, num_frames, channels]
        )

        residual = hidden_states
        hidden_states = self.norm_in(hidden_states)

        hidden_states = self.ff_in(hidden_states)

        if self.is_res:
            hidden_states = hidden_states + residual

        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(norm_hidden_states, encoder_hidden_states=None)
        hidden_states = attn_output + hidden_states

        # 3. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = self.norm2(hidden_states)
            attn_output = self.attn2(
                norm_hidden_states, encoder_hidden_states=encoder_hidden_states
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        ff_output = self.ff(norm_hidden_states)

        if self.is_res:
            hidden_states = ff_output + hidden_states
        else:
            hidden_states = ff_output

        hidden_states = ops.reshape()(
            ops.unsqueeze(0)(hidden_states),
            [batch_size, seq_length, num_frames, channels],
        )
        # hidden_states = hidden_states.permute(0, 2, 1, 3)
        hidden_states = ops.reshape()(
            hidden_states, [batch_size * num_frames, seq_length, channels]
        )

        return hidden_states


class SkipFFTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        kv_input_dim: int,
        kv_input_dim_proj_use_bias: bool,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        attention_out_bias: bool = True,
        dtype: str = "float16",
    ):
        super().__init__()
        if kv_input_dim != dim:
            self.kv_mapper = nn.Linear(
                kv_input_dim, dim, kv_input_dim_proj_use_bias, dtype=dtype
            )
        else:
            self.kv_mapper = None

        self.norm1 = RMSNorm(dim, 1e-06, dtype=dtype)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            out_bias=attention_out_bias,
            dtype=dtype,
        )

        self.norm2 = RMSNorm(dim, 1e-06, dtype=dtype)

        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            out_bias=attention_out_bias,
            dtype=dtype,
        )

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        cross_attention_kwargs: Dict[str, Tensor],
    ):
        cross_attention_kwargs = (
            cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        )

        if self.kv_mapper is not None:
            encoder_hidden_states = self.kv_mapper(ops.silu(encoder_hidden_states))

        norm_hidden_states = self.norm1(hidden_states)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            **cross_attention_kwargs,
        )

        hidden_states = attn_output + hidden_states

        norm_hidden_states = self.norm2(hidden_states)

        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            **cross_attention_kwargs,
        )

        hidden_states = attn_output + hidden_states

        return hidden_states


class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
        dtype: str = "float16",
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias, dtype=dtype)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias, dtype=dtype)
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, bias=bias, dtype=dtype)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim, bias=bias, dtype=dtype)

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(nn.Linear(inner_dim, dim_out, bias=bias, dtype=dtype))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: Tensor, *args, **kwargs) -> Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states
