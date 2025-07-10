from typing import Optional, Union

from honey.compiler import ops

from honey.frontend import IntVar, nn, Tensor

from .normalization import FP32LayerNorm, RMSNorm


# TODO: other processors
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


class Attention(nn.Module):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`):
            The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8):
            The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64):
            The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
        upcast_attention (`bool`, *optional*, defaults to False):
            Set to `True` to upcast the attention computation to `float32`.
        upcast_softmax (`bool`, *optional*, defaults to False):
            Set to `True` to upcast the softmax computation to `float32`.
        cross_attention_norm (`str`, *optional*, defaults to `None`):
            The type of normalization to use for the cross attention. Can be `None`, `layer_norm`, or `group_norm`.
        cross_attention_norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups to use for the group norm in the cross attention.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
        norm_num_groups (`int`, *optional*, defaults to `None`):
            The number of groups to use for the group norm in the attention.
        spatial_norm_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the spatial normalization.
        out_bias (`bool`, *optional*, defaults to `True`):
            Set to `True` to use a bias in the output linear layer.
        scale_qk (`bool`, *optional*, defaults to `True`):
            Set to `True` to scale the query and key by `1 / sqrt(dim_head)`.
        only_cross_attention (`bool`, *optional*, defaults to `False`):
            Set to `True` to only use cross attention and not added_kv_proj_dim. Can only be set to `True` if
            `added_kv_proj_dim` is not `None`.
        eps (`float`, *optional*, defaults to 1e-5):
            An additional value added to the denominator in group normalization that is used for numerical stability.
        rescale_output_factor (`float`, *optional*, defaults to 1.0):
            A factor to rescale the output by dividing it with this value.
        residual_connection (`bool`, *optional*, defaults to `False`):
            Set to `True` to add the residual connection to the output.
        _from_deprecated_attn_block (`bool`, *optional*, defaults to `False`):
            Set to `True` if the attention block is loaded from a deprecated state dict.
        processor (`AttnProcessor`, *optional*, defaults to `None`):
            The attention processor to use. If `None`, defaults to `AttnProcessor2_0` if `torch 2.x` is used and
            `AttnProcessor` otherwise.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        kv_heads: Optional[int] = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        qk_norm: Optional[str] = None,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block: bool = False,
        processor: Optional["AttnProcessor2_0"] = None,
        out_dim: int = None,
        context_pre_only=None,
        pre_only=False,
        dtype: str = "float16",
    ):
        super().__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.is_cross_attention = cross_attention_dim is not None
        self.cross_attention_dim = (
            cross_attention_dim if cross_attention_dim is not None else query_dim
        )
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.dropout = dropout
        self.fused_projections = False
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only

        # we make use of this private variable to know whether this class is loaded
        # with an deprecated state dict so that we can convert it on the fly
        self._from_deprecated_attn_block = _from_deprecated_attn_block

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0

        self.heads = out_dim // dim_head if out_dim is not None else heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads

        self.added_kv_proj_dim = added_kv_proj_dim
        self.only_cross_attention = only_cross_attention

        if self.added_kv_proj_dim is None and self.only_cross_attention:
            raise ValueError(
                "`only_cross_attention` can only be set to True if `added_kv_proj_dim` is not None. Make sure to set either `only_cross_attention=False` or define `added_kv_proj_dim`."
            )

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(
                num_channels=query_dim,
                num_groups=norm_num_groups,
                eps=eps,
                affine=True,
                dtype=dtype,
            )
        else:
            self.group_norm = None

        if spatial_norm_dim is not None:
            self.spatial_norm = SpatialNorm(
                f_channels=query_dim, zq_channels=spatial_norm_dim, dtype=dtype
            )
        else:
            self.spatial_norm = None

        if qk_norm is None:
            self.norm_q = None
            self.norm_k = None
        elif qk_norm == "layer_norm":
            self.norm_q = nn.LayerNorm(dim_head, eps=eps, dtype=dtype)
            self.norm_k = nn.LayerNorm(dim_head, eps=eps, dtype=dtype)
        elif qk_norm == "fp32_layer_norm":
            self.norm_q = FP32LayerNorm(
                dim_head, elementwise_affine=False, bias=False, eps=eps
            )
            self.norm_k = FP32LayerNorm(
                dim_head, elementwise_affine=False, bias=False, eps=eps
            )
        elif qk_norm == "layer_norm_across_heads":
            # Lumina applys qk norm across all heads
            self.norm_q = nn.LayerNorm(dim_head * heads, eps=eps, dtype=dtype)
            self.norm_k = nn.LayerNorm(dim_head * kv_heads, eps=eps, dtype=dtype)
        elif qk_norm == "rms_norm":
            self.norm_q = RMSNorm(dim_head, eps=eps, dtype=dtype)
            self.norm_k = RMSNorm(dim_head, eps=eps, dtype=dtype)
        else:
            raise ValueError(
                f"unknown qk_norm: {qk_norm}. Should be None or 'layer_norm'"
            )

        if cross_attention_norm is None:
            self.norm_cross = None
        elif cross_attention_norm == "layer_norm":
            self.norm_cross = nn.LayerNorm(self.cross_attention_dim, dtype=dtype)
        elif cross_attention_norm == "group_norm":
            if self.added_kv_proj_dim is not None:
                # The given `encoder_hidden_states` are initially of shape
                # (batch_size, seq_len, added_kv_proj_dim) before being projected
                # to (batch_size, seq_len, cross_attention_dim). The norm is applied
                # before the projection, so we need to use `added_kv_proj_dim` as
                # the number of channels for the group norm.
                norm_cross_num_channels = added_kv_proj_dim
            else:
                norm_cross_num_channels = self.cross_attention_dim

            self.norm_cross = nn.GroupNorm(
                num_channels=norm_cross_num_channels,
                num_groups=cross_attention_norm_num_groups,
                eps=1e-5,
                affine=True,
                dtype=dtype,
            )
        else:
            raise ValueError(
                f"unknown cross_attention_norm: {cross_attention_norm}. Should be None, 'layer_norm' or 'group_norm'"
            )

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias, dtype=dtype)

        if not self.only_cross_attention:
            # only relevant for the `AddedKVProcessor` classes
            self.to_k = nn.Linear(
                self.cross_attention_dim, self.inner_dim, bias=bias, dtype=dtype
            )
            self.to_v = nn.Linear(
                self.cross_attention_dim, self.inner_dim, bias=bias, dtype=dtype
            )
        else:
            self.to_k = None
            self.to_v = None

        # to_out cutlass error if these are cast from float8 🤷‍♂️
        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(
                added_kv_proj_dim, self.inner_dim, dtype=dtype if "float8" not in dtype else "float16",
            )
            self.add_v_proj = nn.Linear(
                added_kv_proj_dim, self.inner_dim, dtype=dtype if "float8" not in dtype else "float16",
            )
            if self.context_pre_only is not None:
                self.add_q_proj = nn.Linear(
                    added_kv_proj_dim, self.inner_dim, dtype=dtype if "float8" not in dtype else "float16",
                )

        if not self.pre_only:
            self.to_out = nn.ModuleList([])
            self.to_out.append(
                nn.Linear(self.inner_dim, self.out_dim, bias=out_bias, dtype=dtype)
            )
            self.to_out.append(nn.Dropout(dropout))

        if self.context_pre_only is not None and not self.context_pre_only:
            self.to_add_out = nn.Linear(
                self.inner_dim, self.out_dim, bias=out_bias, dtype=dtype
            )

        if qk_norm is not None and added_kv_proj_dim is not None:
            if qk_norm == "fp32_layer_norm":
                self.norm_added_q = FP32LayerNorm(
                    dim_head, elementwise_affine=False, bias=False, eps=eps
                )
                self.norm_added_k = FP32LayerNorm(
                    dim_head, elementwise_affine=False, bias=False, eps=eps
                )
            elif qk_norm == "rms_norm":
                self.norm_added_q = RMSNorm(dim_head, eps=eps, dtype=dtype)
                self.norm_added_k = RMSNorm(dim_head, eps=eps, dtype=dtype)
        else:
            self.norm_added_q = None
            self.norm_added_k = None
        if processor is None:
            processor = AttnProcessor2_0()
        self.set_processor(processor)

    # def set_attention_slice(self, slice_size: int) -> None:
    #     r"""
    #     Set the slice size for attention computation.

    #     Args:
    #         slice_size (`int`):
    #             The slice size for attention computation.
    #     """
    #     if slice_size is not None and slice_size > self.sliceable_head_dim:
    #         raise ValueError(f"slice_size {slice_size} has to be smaller or equal to {self.sliceable_head_dim}.")

    #     if slice_size is not None and self.added_kv_proj_dim is not None:
    #         processor = SlicedAttnAddedKVProcessor(slice_size)
    #     elif slice_size is not None:
    #         processor = SlicedAttnProcessor(slice_size)
    #     elif self.added_kv_proj_dim is not None:
    #         processor = AttnAddedKVProcessor2_0()
    #     else:
    #         # set attention processor
    #         processor = AttnProcessor2_0()

    #     self.set_processor(processor)

    def set_processor(self, processor: "AttnProcessor2_0") -> None:
        r"""
        Set the attention processor to use.

        Args:
            processor (`AttnProcessor`):
                The attention processor to use.
        """

        self.processor = processor

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        **cross_attention_kwargs,
    ) -> Tensor:
        r"""
        The forward method of the `Attention` class.

        Args:
            hidden_states (`Tensor`):
                The hidden states of the query.
            encoder_hidden_states (`Tensor`, *optional*):
                The hidden states of the encoder.
            attention_mask (`Tensor`, *optional*):
                The attention mask to use. If `None`, no mask is applied.
            **cross_attention_kwargs:
                Additional keyword arguments to pass along to the cross attention.

        Returns:
            `Tensor`: The output of the attention layer.
        """
        # The `Attention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty

        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

    # TODO
    # def batch_to_head_dim(self, tensor: Tensor) -> Tensor:
    #     r"""
    #     Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size // heads, seq_len, dim * heads]`. `heads`
    #     is the number of heads initialized while constructing the `Attention` class.

    #     Args:
    #         tensor (`Tensor`): The tensor to reshape.

    #     Returns:
    #         `Tensor`: The reshaped tensor.
    #     """
    #     head_size = self.heads
    #     batch_size, seq_len, dim = tensor.shape
    #     tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
    #     tensor = tensor.permute(0, 2, 1, 3).reshape(
    #         batch_size // head_size, seq_len, dim * head_size
    #     )
    #     return tensor

    # TODO
    # def head_to_batch_dim(self, tensor: Tensor, out_dim: int = 3) -> Tensor:
    #     r"""
    #     Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size, seq_len, heads, dim // heads]` `heads` is
    #     the number of heads initialized while constructing the `Attention` class.

    #     Args:
    #         tensor (`Tensor`): The tensor to reshape.
    #         out_dim (`int`, *optional*, defaults to `3`): The output dimension of the tensor. If `3`, the tensor is
    #             reshaped to `[batch_size * heads, seq_len, dim // heads]`.

    #     Returns:
    #         `Tensor`: The reshaped tensor.
    #     """
    #     head_size = self.heads
    #     if tensor.ndim == 3:
    #         batch_size, seq_len, dim = tensor.shape
    #         extra_dim = 1
    #     else:
    #         batch_size, extra_dim, seq_len, dim = tensor.shape
    #     tensor = tensor.reshape(
    #         batch_size, seq_len * extra_dim, head_size, dim // head_size
    #     )
    #     tensor = tensor.permute(0, 2, 1, 3)

    #     if out_dim == 3:
    #         tensor = tensor.reshape(
    #             batch_size * head_size, seq_len * extra_dim, dim // head_size
    #         )

    #     return tensor

    # TODO
    # def get_attention_scores(
    #     self,
    #     query: Tensor,
    #     key: Tensor,
    #     attention_mask: Tensor = None,
    # ) -> Tensor:
    #     r"""
    #     Compute the attention scores.

    #     Args:
    #         query (`Tensor`): The query tensor.
    #         key (`Tensor`): The key tensor.
    #         attention_mask (`Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

    #     Returns:
    #         `Tensor`: The attention probabilities/scores.
    #     """
    #     dtype = query.dtype
    #     if self.upcast_attention:
    #         query = query.float()
    #         key = key.float()

    #     if attention_mask is None:
    #         baddbmm_input = torch.empty(
    #             query.shape[0],
    #             query.shape[1],
    #             key.shape[1],
    #             dtype=query.dtype,
    #             device=query.device,
    #         )
    #         beta = 0
    #     else:
    #         baddbmm_input = attention_mask
    #         beta = 1

    #     attention_scores = torch.baddbmm(
    #         baddbmm_input,
    #         query,
    #         key.transpose(-1, -2),
    #         beta=beta,
    #         alpha=self.scale,
    #     )
    #     del baddbmm_input

    #     if self.upcast_softmax:
    #         attention_scores = attention_scores.float()

    #     attention_probs = attention_scores.softmax(dim=-1)
    #     del attention_scores

    #     attention_probs = attention_probs.to(dtype)

    #     return attention_probs

    def prepare_attention_mask(
        self,
        attention_mask: Tensor,
        target_length: int,
        batch_size: int,
        out_dim: int = 3,
    ) -> Tensor:
        r"""
        Prepare the attention mask for the attention computation.

        Args:
            attention_mask (`Tensor`):
                The attention mask to prepare.
            target_length (`int`):
                The target length of the attention mask. This is the length of the attention mask after padding.
            batch_size (`int`):
                The batch size, which is used to repeat the attention mask.
            out_dim (`int`, *optional*, defaults to `3`):
                The output dimension of the attention mask. Can be either `3` or `4`.

        Returns:
            `Tensor`: The prepared attention mask.
        """
        head_size = self.heads
        if attention_mask is None:
            return attention_mask

        batch, current_length = ops.size()(attention_mask)
        if current_length != target_length:
            # TODO: for pipelines such as stable-diffusion, padding cross-attn mask:
            #       we want to instead pad by (0, remaining_length), where remaining_length is:
            #       remaining_length: int = target_length - current_length
            # TODO: re-enable tests/models/test_models_unet_2d_condition.py#test_model_xattn_padding
            attention_mask = ops.concatenate()(
                attention_mask,
                ops.full()([batch, target_length - current_length], fill_value=0.0),
                dim=-1,
            )

        if out_dim == 3:
            if ops.size()(attention_mask, dim=0) < batch_size * head_size:
                attention_mask = ops.repeat_interleave(head_size, 0)(attention_mask)
        elif out_dim == 4:
            attention_mask = ops.unsqueeze(1)(attention_mask)
            attention_mask = ops.repeat_interleave(head_size, 1)(attention_mask)

        return attention_mask

    def norm_encoder_hidden_states(self, encoder_hidden_states: Tensor) -> Tensor:
        r"""
        Normalize the encoder hidden states. Requires `self.norm_cross` to be specified when constructing the
        `Attention` class.

        Args:
            encoder_hidden_states (`Tensor`): Hidden states of the encoder.

        Returns:
            `Tensor`: The normalized encoder hidden states.
        """
        assert (
            self.norm_cross is not None
        ), "self.norm_cross must be defined to call self.norm_encoder_hidden_states"

        if isinstance(self.norm_cross, nn.LayerNorm):
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
        elif isinstance(self.norm_cross, nn.GroupNorm):
            # Group norm norms along the channels dimension and expects
            # input to be in the shape of (N, C, *). In this case, we want
            # to norm along the hidden dimension, so we need to move
            # (batch_size, sequence_length, hidden_size) ->
            # (batch_size, hidden_size, sequence_length)
            encoder_hidden_states = ops.permute021()(encoder_hidden_states)
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
            encoder_hidden_states = ops.permute021()(encoder_hidden_states)
        else:
            assert False

        return encoder_hidden_states


class AttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        temb: Optional[Tensor] = None,
        temperature: Optional[float] = None,
        *args,
        **kwargs,
    ) -> Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = len(ops.size()(hidden_states))

        if input_ndim == 4:
            batch_size, height, width, channel = ops.size()(hidden_states)
            hidden_states = ops.reshape()(
                hidden_states, [batch_size, height * width, channel]
            )

        batch_size, sequence_length, _ = (
            ops.size()(hidden_states)
            if encoder_hidden_states is None
            else ops.size()(encoder_hidden_states)
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = ops.reshape()(
                attention_mask,
                [batch_size, attn.heads, -1, ops.size()(attention_mask, dim=-1)],
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = ops.size()(key, dim=-1)._attrs["int_var"]
        head_dim = inner_dim / attn.heads

        query = ops.permute()(
            ops.reshape()(query, [batch_size, -1, attn.heads, head_dim]), [0, 2, 1, 3]
        )

        key = ops.permute()(
            ops.reshape()(key, [batch_size, -1, attn.heads, head_dim]), [0, 2, 1, 3]
        )
        value = ops.permute()(
            ops.reshape()(value, [batch_size, -1, attn.heads, head_dim]), [0, 2, 1, 3]
        )

        attn_op = ops.mem_eff_attention(causal=False)
        hidden_states = attn_op(query, key, value)

        hidden_states = ops.reshape()(
            hidden_states, [batch_size, -1, attn.heads * head_dim]
        )
        hidden_states = ops.cast()(hidden_states, dtype=query.dtype())

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = ops.reshape()(
                hidden_states, [batch_size, height, width, channel]
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        # if attn.rescale_output_factor != 1.0:
        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor):
    xq_ = ops.reshape()(ops.cast()(xq, "float32"), [*ops.size()(xq)[:-1], -1, 1, 2])
    xk_ = ops.reshape()(ops.cast()(xk, "float32"), [*ops.size()(xk)[:-1], -1, 1, 2])
    freqs_cis_ndim = len(ops.size()(freqs_cis))
    xq_ndim = len(ops.size()(xq_))
    xk_ndim = len(ops.size()(xk_))
    freqs_cis_0 = ops.dynamic_slice()(
        freqs_cis,
        start_indices=[0] * (freqs_cis_ndim - 1) + [0],
        end_indices=[None] * (freqs_cis_ndim - 1) + [1],
    )
    freqs_cis_1 = ops.dynamic_slice()(
        freqs_cis,
        start_indices=[0] * (freqs_cis_ndim - 1) + [1],
        end_indices=[None] * (freqs_cis_ndim - 1) + [2],
    )
    xq_0 = ops.dynamic_slice()(
        xq_,
        start_indices=[0] * (xq_ndim - 1) + [0],
        end_indices=[None] * (xq_ndim - 1) + [1],
    )
    xq_1 = ops.dynamic_slice()(
        xq_,
        start_indices=[0] * (xq_ndim - 1) + [1],
        end_indices=[None] * (xq_ndim - 1) + [2],
    )
    xk_0 = ops.dynamic_slice()(
        xk_,
        start_indices=[0] * (xk_ndim - 1) + [0],
        end_indices=[None] * (xk_ndim - 1) + [1],
    )
    xk_1 = ops.dynamic_slice()(
        xk_,
        start_indices=[0] * (xk_ndim - 1) + [1],
        end_indices=[None] * (xk_ndim - 1) + [2],
    )
    xq_out = freqs_cis_0 * xq_0 + freqs_cis_1 * xq_1
    xk_out = freqs_cis_0 * xk_0 + freqs_cis_1 * xk_1
    return (
        ops.cast()(ops.reshape()(xq_out, ops.size()(xq)), xq.dtype()),
        ops.cast()(ops.reshape()(xk_out, ops.size()(xk)), xk.dtype()),
    )


class FluxSingleAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        image_rotary_emb: Optional[Tensor] = None,
    ) -> Tensor:
        input_ndim = len(ops.size()(hidden_states))

        if input_ndim == 4:
            batch_size, height, width, channel = ops.size()(hidden_states)
            hidden_states = ops.reshape()(
                hidden_states, [batch_size, height * width, channel]
            )

        batch_size = (
            ops.size()(hidden_states, dim=0)
            if encoder_hidden_states is None
            else ops.size()(hidden_states, dim=0)
        )

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = ops.size()(key, dim=-1)._attrs["int_var"]
        head_dim = inner_dim / attn.heads

        query = ops.permute()(
            ops.reshape()(query, [batch_size, -1, attn.heads, head_dim]), [0, 2, 1, 3]
        )

        key = ops.permute()(
            ops.reshape()(key, [batch_size, -1, attn.heads, head_dim]), [0, 2, 1, 3]
        )
        value = ops.permute()(
            ops.reshape()(value, [batch_size, -1, attn.heads, head_dim]), [0, 2, 1, 3]
        )

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            # YiYi to-do: update uising apply_rotary_emb
            # from ..embeddings import apply_rotary_emb
            # query = apply_rotary_emb(query, image_rotary_emb)
            # key = apply_rotary_emb(key, image_rotary_emb)
            query, key = apply_rope(query, key, image_rotary_emb)

        attn_op = ops.mem_eff_attention(causal=False)
        hidden_states = attn_op(query, key, value)

        hidden_states = ops.reshape()(
            hidden_states, [batch_size, -1, attn.heads * head_dim]
        )
        hidden_states = ops.cast()(hidden_states, dtype=query.dtype())

        if input_ndim == 4:
            hidden_states = ops.reshape()(
                hidden_states, [batch_size, height, width, channel]
            )

        return hidden_states


class FluxAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __call__(
        self,
        attn: Attention,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor = None,
        attention_mask: Optional[Tensor] = None,
        image_rotary_emb: Optional[Tensor] = None,
    ) -> Tensor:
        input_ndim = len(ops.size()(hidden_states))

        if input_ndim == 4:
            batch_size, height, width, channel = ops.size()(hidden_states)
            hidden_states = ops.reshape()(
                hidden_states, [batch_size, height * width, channel]
            )
        context_input_ndim = len(ops.size()(encoder_hidden_states))
        if context_input_ndim == 4:
            batch_size, height, width, channel = ops.size()(encoder_hidden_states)
            encoder_hidden_states = ops.reshape()(
                encoder_hidden_states, [batch_size, height * width, channel]
            )

        batch_size = ops.size()(encoder_hidden_states, dim=0)

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = ops.size()(key, dim=-1)._attrs["int_var"]
        head_dim = inner_dim / attn.heads

        query = ops.permute()(
            ops.reshape()(query, [batch_size, -1, attn.heads, head_dim]), [0, 2, 1, 3]
        )

        key = ops.permute()(
            ops.reshape()(key, [batch_size, -1, attn.heads, head_dim]), [0, 2, 1, 3]
        )
        value = ops.permute()(
            ops.reshape()(value, [batch_size, -1, attn.heads, head_dim]), [0, 2, 1, 3]
        )

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = ops.permute()(
            ops.reshape()(
                encoder_hidden_states_query_proj, [batch_size, -1, attn.heads, head_dim]
            ), [0, 2, 1, 3]
        )

        encoder_hidden_states_key_proj = ops.permute()(
            ops.reshape()(
                encoder_hidden_states_key_proj, [batch_size, -1, attn.heads, head_dim]
            ), [0, 2, 1, 3]
        )
        encoder_hidden_states_value_proj = ops.permute()(
            ops.reshape()(
                encoder_hidden_states_value_proj, [batch_size, -1, attn.heads, head_dim]
            ), [0, 2, 1, 3]
        )

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(
                encoder_hidden_states_query_proj
            )
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(
                encoder_hidden_states_key_proj
            )

        # attention
        query = ops.concatenate()([encoder_hidden_states_query_proj, query], dim=2)
        key = ops.concatenate()([encoder_hidden_states_key_proj, key], dim=2)
        value = ops.concatenate()([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            # YiYi to-do: update uising apply_rotary_emb
            # from ..embeddings import apply_rotary_emb
            # query = apply_rotary_emb(query, image_rotary_emb)
            # key = apply_rotary_emb(key, image_rotary_emb)
            query, key = apply_rope(query, key, image_rotary_emb)

        attn_op = ops.mem_eff_attention(causal=False)
        hidden_states = attn_op(query, key, value)

        hidden_states = ops.reshape()(
            hidden_states, [batch_size, -1, attn.heads * head_dim]
        )
        hidden_states = ops.cast()(hidden_states, dtype=query.dtype())

        encoder_hidden_states_dim = ops.size()(encoder_hidden_states, dim=1)._attrs[
            "int_var"
        ]
        hidden_states_dim = ops.size()(hidden_states, dim=1)._attrs["int_var"]
        # TODO: other arange direct dtype
        encoder_hidden_states_indices = ops.arange(
            0, encoder_hidden_states_dim, 1, "int64"
        )()

        hidden_states_indices = ops.arange(
            encoder_hidden_states_dim, hidden_states_dim, 1, "int64"
        )()

        encoder_hidden_states = ops.index_select(dim=1)(
            hidden_states, encoder_hidden_states_indices
        )
        hidden_states = ops.index_select(dim=1)(hidden_states, hidden_states_indices)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if input_ndim == 4:
            hidden_states = ops.reshape()(
                hidden_states, [batch_size, height, width, channel]
            )
        if context_input_ndim == 4:
            encoder_hidden_states = ops.reshape()(
                encoder_hidden_states, [batch_size, height, width, channel]
            )

        return hidden_states, encoder_hidden_states


class JointAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __call__(
        self,
        attn: Attention,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor = None,
        attention_mask: Optional[Tensor] = None,
        *args,
        **kwargs,
    ) -> Tensor:
        residual = hidden_states
        residual_dim = ops.size()(residual, dim=1)._attrs["int_var"]

        input_ndim = len(ops.size()(hidden_states))

        if input_ndim == 4:
            batch_size, height, width, channel = ops.size()(hidden_states)
            hidden_states = ops.reshape()(
                hidden_states, [batch_size, height * width, channel]
            )

        context_input_ndim = len(ops.size()(encoder_hidden_states))
        if context_input_ndim == 4:
            batch_size, height, width, channel = ops.size()(encoder_hidden_states)
            encoder_hidden_states = ops.reshape()(
                encoder_hidden_states, [batch_size, height * width, channel]
            )

        batch_size = ops.size()(encoder_hidden_states, dim=0)

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        # attention
        query = ops.concatenate()([query, encoder_hidden_states_query_proj], dim=1)
        key = ops.concatenate()([key, encoder_hidden_states_key_proj], dim=1)
        value = ops.concatenate()([value, encoder_hidden_states_value_proj], dim=1)

        inner_dim = ops.size()(key, dim=-1)._attrs["int_var"]
        head_dim = inner_dim / attn.heads
        query = ops.permute()(
            ops.reshape()(query, [batch_size, -1, attn.heads, head_dim]), [0, 2, 1, 3]
        )

        key = ops.permute()(
            ops.reshape()(key, [batch_size, -1, attn.heads, head_dim]), [0, 2, 1, 3]
        )
        value = ops.permute()(
            ops.reshape()(value, [batch_size, -1, attn.heads, head_dim]), [0, 2, 1, 3]
        )
        attn_op = ops.mem_eff_attention(causal=False)
        hidden_states = attn_op(query, key, value)

        hidden_states = ops.reshape()(
            hidden_states, [batch_size, -1, attn.heads * head_dim]
        )
        hidden_states = ops.cast()(hidden_states, dtype=query.dtype())

        # Split the attention outputs.

        hidden_states_dim = ops.size()(hidden_states, dim=1)._attrs["int_var"]
        hidden_states_indices = ops.cast()(ops.arange(0, residual_dim, 1)(), "int64")
        encoder_hidden_states_indices = ops.cast()(
            ops.arange(residual_dim, hidden_states_dim, 1)(), "int64"
        )

        hidden_states_sliced = ops.index_select(dim=1)(
            hidden_states, hidden_states_indices
        )
        encoder_hidden_states = ops.index_select(dim=1)(
            hidden_states, encoder_hidden_states_indices
        )
        hidden_states = hidden_states_sliced

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if input_ndim == 4:
            hidden_states = ops.reshape()(
                hidden_states, [batch_size, height, width, channel]
            )
        if context_input_ndim == 4:
            encoder_hidden_states = ops.reshape()(
                encoder_hidden_states, [batch_size, height, width, channel]
            )

        return hidden_states, encoder_hidden_states


# class AttnAddedKVProcessor2_0:
#     r"""
#     Processor for performing scaled dot-product attention (enabled by default if you're using PyTorch 2.0), with extra
#     learnable key and value matrices for the text encoder.
#     """

#     def __call__(
#         self,
#         attn: Attention,
#         hidden_states: Tensor,
#         encoder_hidden_states: Optional[Tensor] = None,
#         attention_mask: Optional[Tensor] = None,
#         *args,
#         **kwargs,
#     ) -> Tensor:
#         residual = hidden_states

#         hidden_states = hidden_states.view(
#             hidden_states.shape[0], hidden_states.shape[1], -1
#         ).transpose(1, 2)
#         batch_size, sequence_length, _ = hidden_states.shape

#         attention_mask = attn.prepare_attention_mask(
#             attention_mask, sequence_length, batch_size, out_dim=4
#         )

#         if encoder_hidden_states is None:
#             encoder_hidden_states = hidden_states
#         elif attn.norm_cross:
#             encoder_hidden_states = attn.norm_encoder_hidden_states(
#                 encoder_hidden_states
#             )

#         hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

#         query = attn.to_q(hidden_states)
#         query = attn.head_to_batch_dim(query, out_dim=4)

#         encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
#         encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
#         encoder_hidden_states_key_proj = attn.head_to_batch_dim(
#             encoder_hidden_states_key_proj, out_dim=4
#         )
#         encoder_hidden_states_value_proj = attn.head_to_batch_dim(
#             encoder_hidden_states_value_proj, out_dim=4
#         )

#         if not attn.only_cross_attention:
#             key = attn.to_k(hidden_states)
#             value = attn.to_v(hidden_states)
#             key = attn.head_to_batch_dim(key, out_dim=4)
#             value = attn.head_to_batch_dim(value, out_dim=4)
#             key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
#             value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
#         else:
#             key = encoder_hidden_states_key_proj
#             value = encoder_hidden_states_value_proj

#         # the output of sdp = (batch, num_heads, seq_len, head_dim)
#         # TODO: add support for attn.scale when we move to Torch 2.1
#         hidden_states = F.scaled_dot_product_attention(
#             query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
#         )
#         hidden_states = hidden_states.transpose(1, 2).reshape(
#             batch_size, -1, residual.shape[1]
#         )

#         # linear proj
#         hidden_states = attn.to_out[0](hidden_states)
#         # dropout
#         hidden_states = attn.to_out[1](hidden_states)

#         hidden_states = hidden_states.transpose(-1, -2).reshape(residual.shape)
#         hidden_states = hidden_states + residual

#         return hidden_states


# class CustomDiffusionAttnProcessor2_0(nn.Module):
#     r"""
#     Processor for implementing attention for the Custom Diffusion method using PyTorch 2.0’s memory-efficient scaled
#     dot-product attention.

#     Args:
#         train_kv (`bool`, defaults to `True`):
#             Whether to newly train the key and value matrices corresponding to the text features.
#         train_q_out (`bool`, defaults to `True`):
#             Whether to newly train query matrices corresponding to the latent image features.
#         hidden_size (`int`, *optional*, defaults to `None`):
#             The hidden size of the attention layer.
#         cross_attention_dim (`int`, *optional*, defaults to `None`):
#             The number of channels in the `encoder_hidden_states`.
#         out_bias (`bool`, defaults to `True`):
#             Whether to include the bias parameter in `train_q_out`.
#         dropout (`float`, *optional*, defaults to 0.0):
#             The dropout probability to use.
#     """

#     def __init__(
#         self,
#         train_kv: bool = True,
#         train_q_out: bool = True,
#         hidden_size: Optional[int] = None,
#         cross_attention_dim: Optional[int] = None,
#         out_bias: bool = True,
#         dropout: float = 0.0,
#     ):
#         super().__init__()
#         self.train_kv = train_kv
#         self.train_q_out = train_q_out

#         self.hidden_size = hidden_size
#         self.cross_attention_dim = cross_attention_dim

#         # `_custom_diffusion` id for easy serialization and loading.
#         if self.train_kv:
#             self.to_k_custom_diffusion = nn.Linear(
#                 cross_attention_dim or hidden_size, hidden_size, bias=False
#             )
#             self.to_v_custom_diffusion = nn.Linear(
#                 cross_attention_dim or hidden_size, hidden_size, bias=False
#             )
#         if self.train_q_out:
#             self.to_q_custom_diffusion = nn.Linear(hidden_size, hidden_size, bias=False)
#             self.to_out_custom_diffusion = nn.ModuleList([])
#             self.to_out_custom_diffusion.append(
#                 nn.Linear(hidden_size, hidden_size, bias=out_bias)
#             )
#             self.to_out_custom_diffusion.append(nn.Dropout(dropout))

#     def __call__(
#         self,
#         attn: Attention,
#         hidden_states: Tensor,
#         encoder_hidden_states: Optional[Tensor] = None,
#         attention_mask: Optional[Tensor] = None,
#     ) -> Tensor:
#         batch_size, sequence_length, _ = hidden_states.shape
#         attention_mask = attn.prepare_attention_mask(
#             attention_mask, sequence_length, batch_size
#         )
#         if self.train_q_out:
#             query = self.to_q_custom_diffusion(hidden_states)
#         else:
#             query = attn.to_q(hidden_states)

#         if encoder_hidden_states is None:
#             crossattn = False
#             encoder_hidden_states = hidden_states
#         else:
#             crossattn = True
#             if attn.norm_cross:
#                 encoder_hidden_states = attn.norm_encoder_hidden_states(
#                     encoder_hidden_states
#                 )

#         if self.train_kv:
#             key = self.to_k_custom_diffusion(
#                 encoder_hidden_states.to(self.to_k_custom_diffusion.weight.dtype)
#             )
#             value = self.to_v_custom_diffusion(
#                 encoder_hidden_states.to(self.to_v_custom_diffusion.weight.dtype)
#             )
#             key = key.to(attn.to_q.weight.dtype)
#             value = value.to(attn.to_q.weight.dtype)

#         else:
#             key = attn.to_k(encoder_hidden_states)
#             value = attn.to_v(encoder_hidden_states)

#         if crossattn:
#             detach = torch.ones_like(key)
#             detach[:, :1, :] = detach[:, :1, :] * 0.0
#             key = detach * key + (1 - detach) * key.detach()
#             value = detach * value + (1 - detach) * value.detach()

#         inner_dim = hidden_states.shape[-1]

#         head_dim = inner_dim // attn.heads
#         query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
#         key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
#         value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

#         # the output of sdp = (batch, num_heads, seq_len, head_dim)
#         # TODO: add support for attn.scale when we move to Torch 2.1
#         hidden_states = F.scaled_dot_product_attention(
#             query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
#         )

#         hidden_states = hidden_states.transpose(1, 2).reshape(
#             batch_size, -1, attn.heads * head_dim
#         )
#         hidden_states = hidden_states.to(query.dtype)

#         if self.train_q_out:
#             # linear proj
#             hidden_states = self.to_out_custom_diffusion[0](hidden_states)
#             # dropout
#             hidden_states = self.to_out_custom_diffusion[1](hidden_states)
#         else:
#             # linear proj
#             hidden_states = attn.to_out[0](hidden_states)
#             # dropout
#             hidden_states = attn.to_out[1](hidden_states)

#         return hidden_states


# class SlicedAttnProcessor:
#     r"""
#     Processor for implementing sliced attention.

#     Args:
#         slice_size (`int`, *optional*):
#             The number of steps to compute attention. Uses as many slices as `attention_head_dim // slice_size`, and
#             `attention_head_dim` must be a multiple of the `slice_size`.
#     """

#     def __init__(self, slice_size: int):
#         self.slice_size = slice_size

#     def __call__(
#         self,
#         attn: Attention,
#         hidden_states: Tensor,
#         encoder_hidden_states: Optional[Tensor] = None,
#         attention_mask: Optional[Tensor] = None,
#     ) -> Tensor:
#         residual = hidden_states

#         input_ndim = hidden_states.ndim

#         if input_ndim == 4:
#             batch_size, channel, height, width = hidden_states.shape
#             hidden_states = hidden_states.view(
#                 batch_size, channel, height * width
#             ).transpose(1, 2)

#         batch_size, sequence_length, _ = (
#             hidden_states.shape
#             if encoder_hidden_states is None
#             else encoder_hidden_states.shape
#         )
#         attention_mask = attn.prepare_attention_mask(
#             attention_mask, sequence_length, batch_size
#         )

#         if attn.group_norm is not None:
#             hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
#                 1, 2
#             )

#         query = attn.to_q(hidden_states)
#         dim = query.shape[-1]
#         query = attn.head_to_batch_dim(query)

#         if encoder_hidden_states is None:
#             encoder_hidden_states = hidden_states
#         elif attn.norm_cross:
#             encoder_hidden_states = attn.norm_encoder_hidden_states(
#                 encoder_hidden_states
#             )

#         key = attn.to_k(encoder_hidden_states)
#         value = attn.to_v(encoder_hidden_states)
#         key = attn.head_to_batch_dim(key)
#         value = attn.head_to_batch_dim(value)

#         batch_size_attention, query_tokens, _ = query.shape
#         hidden_states = torch.zeros(
#             (batch_size_attention, query_tokens, dim // attn.heads),
#             device=query.device,
#             dtype=query.dtype,
#         )

#         for i in range(batch_size_attention // self.slice_size):
#             start_idx = i * self.slice_size
#             end_idx = (i + 1) * self.slice_size

#             query_slice = query[start_idx:end_idx]
#             key_slice = key[start_idx:end_idx]
#             attn_mask_slice = (
#                 attention_mask[start_idx:end_idx]
#                 if attention_mask is not None
#                 else None
#             )

#             attn_slice = attn.get_attention_scores(
#                 query_slice, key_slice, attn_mask_slice
#             )

#             attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

#             hidden_states[start_idx:end_idx] = attn_slice

#         hidden_states = attn.batch_to_head_dim(hidden_states)

#         # linear proj
#         hidden_states = attn.to_out[0](hidden_states)
#         # dropout
#         hidden_states = attn.to_out[1](hidden_states)

#         if input_ndim == 4:
#             hidden_states = hidden_states.transpose(-1, -2).reshape(
#                 batch_size, channel, height, width
#             )

#         if attn.residual_connection:
#             hidden_states = hidden_states + residual

#         hidden_states = hidden_states / attn.rescale_output_factor

#         return hidden_states


# class SlicedAttnAddedKVProcessor:
#     r"""
#     Processor for implementing sliced attention with extra learnable key and value matrices for the text encoder.

#     Args:
#         slice_size (`int`, *optional*):
#             The number of steps to compute attention. Uses as many slices as `attention_head_dim // slice_size`, and
#             `attention_head_dim` must be a multiple of the `slice_size`.
#     """

#     def __init__(self, slice_size):
#         self.slice_size = slice_size

#     def __call__(
#         self,
#         attn: "Attention",
#         hidden_states: Tensor,
#         encoder_hidden_states: Optional[Tensor] = None,
#         attention_mask: Optional[Tensor] = None,
#         temb: Optional[Tensor] = None,
#     ) -> Tensor:
#         residual = hidden_states

#         if attn.spatial_norm is not None:
#             hidden_states = attn.spatial_norm(hidden_states, temb)

#         hidden_states = hidden_states.view(
#             hidden_states.shape[0], hidden_states.shape[1], -1
#         ).transpose(1, 2)

#         batch_size, sequence_length, _ = hidden_states.shape

#         attention_mask = attn.prepare_attention_mask(
#             attention_mask, sequence_length, batch_size
#         )

#         if encoder_hidden_states is None:
#             encoder_hidden_states = hidden_states
#         elif attn.norm_cross:
#             encoder_hidden_states = attn.norm_encoder_hidden_states(
#                 encoder_hidden_states
#             )

#         hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

#         query = attn.to_q(hidden_states)
#         dim = query.shape[-1]
#         query = attn.head_to_batch_dim(query)

#         encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
#         encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

#         encoder_hidden_states_key_proj = attn.head_to_batch_dim(
#             encoder_hidden_states_key_proj
#         )
#         encoder_hidden_states_value_proj = attn.head_to_batch_dim(
#             encoder_hidden_states_value_proj
#         )

#         if not attn.only_cross_attention:
#             key = attn.to_k(hidden_states)
#             value = attn.to_v(hidden_states)
#             key = attn.head_to_batch_dim(key)
#             value = attn.head_to_batch_dim(value)
#             key = torch.cat([encoder_hidden_states_key_proj, key], dim=1)
#             value = torch.cat([encoder_hidden_states_value_proj, value], dim=1)
#         else:
#             key = encoder_hidden_states_key_proj
#             value = encoder_hidden_states_value_proj

#         batch_size_attention, query_tokens, _ = query.shape
#         hidden_states = torch.zeros(
#             (batch_size_attention, query_tokens, dim // attn.heads),
#             device=query.device,
#             dtype=query.dtype,
#         )

#         for i in range(batch_size_attention // self.slice_size):
#             start_idx = i * self.slice_size
#             end_idx = (i + 1) * self.slice_size

#             query_slice = query[start_idx:end_idx]
#             key_slice = key[start_idx:end_idx]
#             attn_mask_slice = (
#                 attention_mask[start_idx:end_idx]
#                 if attention_mask is not None
#                 else None
#             )

#             attn_slice = attn.get_attention_scores(
#                 query_slice, key_slice, attn_mask_slice
#             )

#             attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

#             hidden_states[start_idx:end_idx] = attn_slice

#         hidden_states = attn.batch_to_head_dim(hidden_states)

#         # linear proj
#         hidden_states = attn.to_out[0](hidden_states)
#         # dropout
#         hidden_states = attn.to_out[1](hidden_states)

#         hidden_states = hidden_states.transpose(-1, -2).reshape(residual.shape)
#         hidden_states = hidden_states + residual

#         return hidden_states


class SpatialNorm(nn.Module):
    """
    Spatially conditioned normalization as defined in https://arxiv.org/abs/2209.09002.

    Args:
        f_channels (`int`):
            The number of channels for input to group normalization layer, and output of the spatial norm layer.
        zq_channels (`int`):
            The number of channels for the quantized vector as described in the paper.
    """

    def __init__(self, f_channels: int, zq_channels: int, dtype: str = "float16"):
        super().__init__()
        self.norm_layer = nn.GroupNorm(
            num_channels=f_channels,
            num_groups=32,
            eps=1e-6,
            affine=True,
            dtype=dtype,
        )
        self.conv_y = nn.Conv2d(
            zq_channels,
            f_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dtype=dtype,
        )
        self.conv_b = nn.Conv2d(
            zq_channels,
            f_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dtype=dtype,
        )

    def forward(self, f: Tensor, zq: Tensor) -> Tensor:
        _, h, w, _ = ops.size()(f)
        f_size = ops.size()(zq)
        f_size[1] = h
        f_size[2] = w
        f_size = [x._attrs["int_var"] for x in f_size]
        f_size = Tensor(f_size)
        zq = ops.upsampling2d(scale_factor=2.0, mode="nearest")(zq, out=f_size)
        norm_f = self.norm_layer(f)
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        return new_f


LORA_ATTENTION_PROCESSORS = (
    # LoRAAttnProcessor,
    # LoRAAttnProcessor2_0,
    # LoRAXFormersAttnProcessor,
    # LoRAAttnAddedKVProcessor,
)

ADDED_KV_ATTENTION_PROCESSORS = (
    # # AttnAddedKVProcessor,
    # SlicedAttnAddedKVProcessor,
    # AttnAddedKVProcessor2_0,
    # # XFormersAttnAddedKVProcessor,
    # # LoRAAttnAddedKVProcessor,
)

CROSS_ATTENTION_PROCESSORS = (
    # AttnProcessor,
    AttnProcessor2_0,
    # # XFormersAttnProcessor,
    # SlicedAttnProcessor,
    # # LoRAAttnProcessor,
    # LoRAAttnProcessor2_0,
    # LoRAXFormersAttnProcessor,
    # IPAdapterAttnProcessor,
    # IPAdapterAttnProcessor2_0,
)

AttentionProcessor = Union[
    # AttnProcessor,
    AttnProcessor2_0,
    # # FusedAttnProcessor2_0,
    # # XFormersAttnProcessor,
    # SlicedAttnProcessor,
    # # AttnAddedKVProcessor,
    # SlicedAttnAddedKVProcessor,
    # AttnAddedKVProcessor2_0,
    # # XFormersAttnAddedKVProcessor,
    # # CustomDiffusionAttnProcessor,
    # # CustomDiffusionXFormersAttnProcessor,
    # CustomDiffusionAttnProcessor2_0,
    # deprecated
    # LoRAAttnProcessor,
    # LoRAAttnProcessor2_0,
    # LoRAXFormersAttnProcessor,
    # LoRAAttnAddedKVProcessor,
]
