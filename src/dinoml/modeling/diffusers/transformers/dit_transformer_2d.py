from typing import Any, Dict, Optional

from dinoml.compiler import ops

from dinoml.frontend import nn, Tensor

from ..attention import BasicTransformerBlock
from ..embeddings import PatchEmbed
from ..modeling_outputs import Transformer2DModelOutput


class DiTTransformer2DModel(nn.Module):
    r"""
    A 2D Transformer model as introduced in DiT (https://arxiv.org/abs/2212.09748).

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
        attention_bias (bool, optional, defaults to True):
            Configure if the Transformer blocks' attention should contain a bias parameter.
        sample_size (int, defaults to 32):
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
        norm_eps (float, optional, defaults to 1e-5):
            A small constant added to the denominator in normalization layers to prevent division by zero.
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 72,
        in_channels: int = 4,
        out_channels: Optional[int] = None,
        num_layers: int = 28,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        attention_bias: bool = True,
        sample_size: int = 32,
        patch_size: int = 2,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm_zero",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        dtype: str = "float16",
        **kwargs,
    ):
        super().__init__()

        # Validate inputs.
        if norm_type != "ada_norm_zero":
            raise NotImplementedError(
                f"Forward pass is not implemented when `patch_size` is not None and `norm_type` is '{norm_type}'."
            )
        elif norm_type == "ada_norm_zero" and num_embeds_ada_norm is None:
            raise ValueError(
                f"When using a `patch_size` and this `norm_type` ({norm_type}), `num_embeds_ada_norm` cannot be None."
            )

        # Set some common variables used across the board.
        self.attention_head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.out_channels = in_channels if out_channels is None else out_channels
        self.gradient_checkpointing = False

        # 2. Initialize the position embedding and transformer blocks.
        self.height = sample_size
        self.width = sample_size

        self.patch_size = patch_size
        self.pos_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=self.inner_dim,
            dtype=dtype,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        # 3. Output blocks.
        self.norm_out = nn.LayerNorm(
            self.inner_dim, elementwise_affine=False, eps=1e-6, dtype=dtype
        )
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim, dtype=dtype)
        self.proj_out_2 = nn.Linear(
            self.inner_dim, patch_size * patch_size * self.out_channels, dtype=dtype
        )

    def forward(
        self,
        hidden_states: Tensor,
        timestep: Optional[Tensor] = None,
        class_labels: Optional[Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        return_dict: bool = True,
    ):
        """
        The [`DiTTransformer2DModel`] forward method.

        Args:
            hidden_states (`Tensor` of shape `(batch size, num latent pixels)` if discrete, `Tensor` of shape `(batch size, height, width, channel)` if continuous):
                Input `hidden_states`.
            timestep ( `Tensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `Tensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        # 1. Input
        height, width = (
            ops.size()(hidden_states, dim=1)._attrs["int_var"] / self.patch_size,
            ops.size()(hidden_states, dim=2)._attrs["int_var"] / self.patch_size,
        )
        hidden_states = self.pos_embed.forward(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
            )

        # 3. Output
        conditioning = self.transformer_blocks[0].norm1.emb(
            timestep, class_labels, hidden_dtype=hidden_states.dtype()
        )
        shift, scale = ops.chunk()(
            self.proj_out_1(ops.silu(conditioning)), chunks=2, dim=1
        )
        hidden_states = self.norm_out(hidden_states) * (
            1 + ops.unsqueeze(1)(scale)
        ) + ops.unsqueeze(1)(shift)
        hidden_states = self.proj_out_2(hidden_states)

        # unpatchify
        height = width = ops.size()(hidden_states, dim=1) ** 0.5
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
