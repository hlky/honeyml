from typing import Any, Dict, Optional

from dinoml.compiler import ops

from dinoml.frontend import nn, Tensor

from ..attention import BasicTransformerBlock
from ..embeddings import PatchEmbed, PixArtAlphaTextProjection
from ..modeling_outputs import Transformer2DModelOutput

from ..normalization import AdaLayerNormSingle


class PixArtTransformer2DModel(nn.Module):
    r"""
    A 2D Transformer model as introduced in PixArt family of models (https://arxiv.org/abs/2310.00426,
    https://arxiv.org/abs/2403.04692).

    Parameters:
        num_attention_heads (int, optional, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (int, optional, defaults to 72): The number of channels in each head.
        in_channels (int, defaults to 4): The number of channels in the input.
        out_channels (int, optional):
            The number of channels in the output. Specify this parameter if the output channel number differs from the
            input.
        num_layers (int, optional, defaults to 28): The number of layers of Transformer blocks to use.
        dropout (float, optional, defaults to 0.0): The dropout probability to use within the Transformer blocks.
        norm_num_groups (int, optional, defaults to 32):
            Number of groups for group normalization within Transformer blocks.
        cross_attention_dim (int, optional):
            The dimensionality for cross-attention layers, typically matching the encoder's hidden dimension.
        attention_bias (bool, optional, defaults to True):
            Configure if the Transformer blocks' attention should contain a bias parameter.
        sample_size (int, defaults to 128):
            The width of the latent images. This parameter is fixed during training.
        patch_size (int, defaults to 2):
            Size of the patches the model processes, relevant for architectures working on non-sequential data.
        activation_fn (str, optional, defaults to "gelu-approximate"):
            Activation function to use in feed-forward networks within Transformer blocks.
        num_embeds_ada_norm (int, optional, defaults to 1000):
            Number of embeddings for AdaLayerNorm, fixed during training and affects the maximum denoising steps during
            inference.
        upcast_attention (bool, optional, defaults to False):
            If true, upcasts the attention mechanism dimensions for potentially improved performance.
        norm_type (str, optional, defaults to "ada_norm_zero"):
            Specifies the type of normalization used, can be 'ada_norm_zero'.
        norm_elementwise_affine (bool, optional, defaults to False):
            If true, enables element-wise affine parameters in the normalization layers.
        norm_eps (float, optional, defaults to 1e-6):
            A small constant added to the denominator in normalization layers to prevent division by zero.
        interpolation_scale (int, optional): Scale factor to use during interpolating the position embeddings.
        use_additional_conditions (bool, optional): If we're using additional conditions as inputs.
        attention_type (str, optional, defaults to "default"): Kind of attention mechanism to be used.
        caption_channels (int, optional, defaults to None):
            Number of channels to use for projecting the caption embeddings.
        use_linear_projection (bool, optional, defaults to False):
            Deprecated argument. Will be removed in a future version.
        num_vector_embeds (bool, optional, defaults to False):
            Deprecated argument. Will be removed in a future version.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["BasicTransformerBlock", "PatchEmbed"]

    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 72,
        in_channels: int = 4,
        out_channels: Optional[int] = 8,
        num_layers: int = 28,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = 1152,
        attention_bias: bool = True,
        sample_size: int = 128,
        patch_size: int = 2,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm_single",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        interpolation_scale: Optional[int] = None,
        use_additional_conditions: Optional[bool] = None,
        caption_channels: Optional[int] = None,
        attention_type: Optional[str] = "default",
        dtype: str = "float16",
        **kwargs,
    ):
        super().__init__()
        self.patch_size = patch_size

        # Validate inputs.
        if norm_type != "ada_norm_single":
            raise NotImplementedError(
                f"Forward pass is not implemented when `patch_size` is not None and `norm_type` is '{norm_type}'."
            )
        elif norm_type == "ada_norm_single" and num_embeds_ada_norm is None:
            raise ValueError(
                f"When using a `patch_size` and this `norm_type` ({norm_type}), `num_embeds_ada_norm` cannot be None."
            )

        # Set some common variables used across the board.
        self.attention_head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.out_channels = in_channels if out_channels is None else out_channels
        if use_additional_conditions is None:
            if sample_size == 128:
                use_additional_conditions = True
            else:
                use_additional_conditions = False
        self.use_additional_conditions = use_additional_conditions

        self.gradient_checkpointing = False

        # 2. Initialize the position embedding and transformer blocks.
        self.height = sample_size
        self.width = sample_size

        interpolation_scale = (
            interpolation_scale
            if interpolation_scale is not None
            else max(sample_size // 64, 1)
        )
        self.pos_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=self.inner_dim,
            interpolation_scale=interpolation_scale,
            dtype=dtype,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    attention_type=attention_type,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        # 3. Output blocks.
        self.norm_out = nn.LayerNorm(
            self.inner_dim, elementwise_affine=False, eps=1e-6, dtype=dtype
        )
        self.scale_shift_table = nn.Parameter([2, self.inner_dim], dtype=dtype)
        self.proj_out = nn.Linear(
            self.inner_dim, patch_size * patch_size * self.out_channels, dtype=dtype
        )

        self.adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            use_additional_conditions=self.use_additional_conditions,
            dtype=dtype,
        )
        self.caption_projection = None
        if caption_channels is not None:
            self.caption_projection = PixArtAlphaTextProjection(
                in_features=caption_channels, hidden_size=self.inner_dim, dtype=dtype
            )

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        timestep: Optional[Tensor] = None,
        added_cond_kwargs: Dict[str, Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        return_dict: bool = True,
    ):
        """
        The [`PixArtTransformer2DModel`] forward method.

        Args:
            hidden_states (`Tensor` of shape `(batch size, height, width, channel)`):
                Input `hidden_states`.
            encoder_hidden_states (`Tensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep (`Tensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            added_cond_kwargs: (`Dict[str, Any]`, *optional*): Additional conditions to be used as inputs.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            attention_mask ( `Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if self.use_additional_conditions and added_cond_kwargs is None:
            raise ValueError(
                "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
            )

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and len(ops.size()(attention_mask)) == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (
                1 - ops.cast()(attention_mask, dtype=hidden_states.dtype())
            ) * -10000.0
            attention_mask = ops.unsqueeze(1)(attention_mask)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if (
            encoder_attention_mask is not None
            and len(ops.size()(encoder_attention_mask)) == 2
        ):
            encoder_attention_mask = (
                1 - ops.cast()(encoder_attention_mask, dtype=hidden_states.dtype())
            ) * -10000.0
            encoder_attention_mask = ops.unsqueeze(1)(encoder_attention_mask)

        # 1. Input
        batch_size = ops.size()(hidden_states, dim=0)
        height, width = (
            ops.size()(hidden_states, dim=1)._attrs["int_var"] / self.patch_size,
            ops.size()(hidden_states, dim=2)._attrs["int_var"] / self.patch_size,
        )
        hidden_states = self.pos_embed(hidden_states)

        timestep, embedded_timestep = self.adaln_single(
            timestep,
            added_cond_kwargs,
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype(),
        )

        if self.caption_projection is not None:
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = ops.reshape()(
                encoder_hidden_states,
                [batch_size, -1, ops.size()(hidden_states, dim=-1)],
            )

        # 2. Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=None,
            )

        # 3. Output
        shift, scale = ops.chunk()(
            ops.unsqueeze(0)(self.scale_shift_table.tensor())
            + ops.unsqueeze(1)(embedded_timestep),
            chunks=2,
            dim=1,
        )
        hidden_states = self.norm_out(hidden_states)
        # Modulation
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.proj_out(hidden_states)
        hidden_states = ops.squeeze(1)(hidden_states)

        # unpatchify
        hidden_states = ops.reshape()(
            hidden_states,
            [
                -1,
                height,
                width,
                self.patch_size,
                self.patch_size,
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
