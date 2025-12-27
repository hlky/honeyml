from typing import Any, Dict, List, Optional, Union

from dinoml.compiler import ops

from dinoml.frontend import nn, Tensor

from ..attention import JointTransformerBlock
from ..embeddings import CombinedTimestepTextProjEmbeddings, PatchEmbed
from ..normalization import AdaLayerNormContinuous
from .transformer_2d import Transformer2DModelOutput


class SD3Transformer2DModel(nn.Module):
    """
    The Transformer model introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        sample_size (`int`): The width of the latent images. This is fixed during training since
            it is used to learn a number of position embeddings.
        patch_size (`int`): Patch size to turn the input data into small patches.
        in_channels (`int`, *optional*, defaults to 16): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 18): The number of layers of Transformer blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 18): The number of heads to use for multi-head attention.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        caption_projection_dim (`int`): Number of dimensions to use when projecting the `encoder_hidden_states`.
        pooled_projection_dim (`int`): Number of dimensions to use when projecting the `pooled_projections`.
        out_channels (`int`, defaults to 16): Number of output channels.

    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 18,
        attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1152,
        pooled_projection_dim: int = 2048,
        out_channels: int = 16,
        pos_embed_max_size: int = 96,
        dtype: str = "float16",
        **kwargs,
    ):
        super().__init__()
        default_out_channels = in_channels
        self.out_channels = (
            out_channels if out_channels is not None else default_out_channels
        )
        self.inner_dim = num_attention_heads * attention_head_dim
        self.attention_head_dim = attention_head_dim
        self.num_attention_heads = num_attention_heads
        self.joint_attention_dim = joint_attention_dim
        self.caption_projection_dim = caption_projection_dim
        self.pooled_projection_dim = pooled_projection_dim
        self.patch_size = patch_size
        self.sample_size = sample_size

        self.pos_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,  # hard-code for now.
            dtype=dtype,
        )
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim,
            pooled_projection_dim=pooled_projection_dim,
            dtype=dtype,
        )
        self.context_embedder = nn.Linear(
            joint_attention_dim, caption_projection_dim, dtype=dtype
        )

        # `attention_head_dim` is doubled to account for the mixing.
        # It needs to crafted when we get the actual checkpoints.
        self.transformer_blocks = nn.ModuleList(
            [
                JointTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=self.inner_dim,
                    context_pre_only=i == num_layers - 1,
                )
                for i in range(num_layers)
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
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[Tensor, Transformer2DModelOutput]:
        """
        The [`SD3Transformer2DModel`] forward method.

        Args:
            hidden_states (`Tensor` of shape `(batch size, height, width, channel)`):
                Input `hidden_states`.
            encoder_hidden_states (`Tensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`Tensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `Tensor`):
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
        height, width = (
            ops.size()(hidden_states, dim=1)._attrs["int_var"] / self.patch_size,
            ops.size()(hidden_states, dim=2)._attrs["int_var"] / self.patch_size,
        )

        hidden_states = self.pos_embed(hidden_states)
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
            )

            # controlnet residual
            if (
                block_controlnet_hidden_states is not None
                and block.context_pre_only is False
            ):
                interval_control = len(self.transformer_blocks) // len(
                    block_controlnet_hidden_states
                )
                hidden_states = (
                    hidden_states
                    + block_controlnet_hidden_states[index_block // interval_control]
                )

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

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
