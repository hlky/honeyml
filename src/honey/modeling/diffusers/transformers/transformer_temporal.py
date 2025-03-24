from dataclasses import dataclass
from typing import Any, Dict, Optional

from honey.compiler import ops

from honey.frontend import nn, Tensor

from ..attention import BasicTransformerBlock, TemporalBasicTransformerBlock
from ..embeddings import TimestepEmbedding, Timesteps
from ..resnet import AlphaBlender

from ..utils import BaseOutput


@dataclass
class TransformerTemporalModelOutput(BaseOutput):
    """
    The output of [`TransformerTemporalModel`].

    Args:
        sample (`Tensor` of shape `(batch_size x num_frames, height, width, num_channels)`):
            The hidden states output conditioned on `encoder_hidden_states` input.
    """

    sample: Tensor


class TransformerTemporalModel(nn.Module):
    """
    A Transformer model for video-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlock` attention should contain a bias parameter.
        sample_size (`int`, *optional*): The width of the latent images (specify if the input is **discrete**).
            This is fixed during training since it is used to learn a number of position embeddings.
        activation_fn (`str`, *optional*, defaults to `"geglu"`):
            Activation function to use in feed-forward. See `diffusers.models.activations.get_activation` for supported
            activation functions.
        norm_elementwise_affine (`bool`, *optional*):
            Configure if the `TransformerBlock` should use learnable elementwise affine parameters for normalization.
        double_self_attention (`bool`, *optional*):
            Configure if each `TransformerBlock` should contain two self-attention layers.
        positional_embeddings: (`str`, *optional*):
            The type of positional embeddings to apply to the sequence input before passing use.
        num_positional_embeddings: (`int`, *optional*):
            The maximum length of the sequence over which to apply positional embeddings.
    """

    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        activation_fn: str = "geglu",
        norm_elementwise_affine: bool = True,
        double_self_attention: bool = True,
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        dtype: str = "float16",
        **kwargs,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        self.in_channels = in_channels

        self.norm = nn.GroupNorm(
            num_groups=norm_num_groups,
            num_channels=in_channels,
            eps=1e-6,
            affine=True,
            dtype=dtype,
        )
        self.proj_in = nn.Linear(in_channels, inner_dim, dtype=dtype)

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    double_self_attention=double_self_attention,
                    norm_elementwise_affine=norm_elementwise_affine,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=num_positional_embeddings,
                    dtype=dtype,
                )
                for d in range(num_layers)
            ]
        )

        self.proj_out = nn.Linear(inner_dim, in_channels, dtype=dtype)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        timestep: Optional[Tensor] = None,
        class_labels: Tensor = None,
        num_frames: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> TransformerTemporalModelOutput:
        """
        The [`TransformerTemporal`] forward method.

        Args:
            hidden_states (`Tensor` of shape `(batch size, num latent pixels)` if discrete, `Tensor` of shape `(batch size, height, width, channel)` if continuous):
                Input hidden_states.
            encoder_hidden_states ( `Tensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `Tensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `Tensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            num_frames (`int`, *optional*, defaults to 1):
                The number of frames to be processed per batch. This is used to reshape the hidden states.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformers.transformer_temporal.TransformerTemporalModelOutput`]
                instead of a plain tuple.

        Returns:
            [`~models.transformers.transformer_temporal.TransformerTemporalModelOutput`] or `tuple`:
                If `return_dict` is True, an
                [`~models.transformers.transformer_temporal.TransformerTemporalModelOutput`] is returned, otherwise a
                `tuple` where the first element is the sample tensor.
        """
        # 1. Input
        batch_frames, height, width, channel = ops.size()(hidden_states)
        batch_size = batch_frames / num_frames

        residual = hidden_states

        hidden_states = ops.reshape()(
            ops.unsqueeze(0)(hidden_states),
            [batch_size, num_frames, height, width, channel],
        )

        hidden_states = self.norm(hidden_states)
        hidden_states = ops.reshape()(
            ops.permute()(hidden_states, [0, 2, 3, 1, 4]),
            [batch_size * height * width, num_frames, channel],
        )

        hidden_states = self.proj_in(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
            )

        # 3. Output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = ops.permute()(
            ops.reshape()(
                ops.unsqueeze(0)(ops.unsqueeze(0)(hidden_states)),
                [batch_size, height, width, num_frames, channel],
            ),
            [0, 3, 1, 2, 4],
        )
        hidden_states = ops.reshape()(
            hidden_states, [batch_frames, height, width, channel]
        )

        output = hidden_states + residual

        if not return_dict:
            return (output,)

        return TransformerTemporalModelOutput(sample=output)


class TransformerSpatioTemporalModel(nn.Module):
    """
    A Transformer model for video-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        out_channels (`int`, *optional*):
            The number of channels in the output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
    """

    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: int = 320,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        cross_attention_dim: Optional[int] = None,
        dtype: str = "float16",
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim

        # 2. Define input layers
        self.in_channels = in_channels
        self.norm = nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, dtype=dtype
        )
        self.proj_in = nn.Linear(in_channels, inner_dim, dtype=dtype)

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    dtype=dtype,
                )
                for d in range(num_layers)
            ]
        )

        time_mix_inner_dim = inner_dim
        self.temporal_transformer_blocks = nn.ModuleList(
            [
                TemporalBasicTransformerBlock(
                    inner_dim,
                    time_mix_inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        time_embed_dim = in_channels * 4
        self.time_pos_embed = TimestepEmbedding(
            in_channels, time_embed_dim, out_dim=in_channels, dtype=dtype
        )
        self.time_proj = Timesteps(in_channels, True, 0, dtype=dtype)
        self.time_mixer = AlphaBlender(
            alpha=0.5, merge_strategy="learned_with_images", dtype=dtype
        )

        # 4. Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        # TODO: should use out_channels for continuous projections
        self.proj_out = nn.Linear(inner_dim, in_channels, dtype=dtype)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        image_only_indicator: Optional[Tensor] = None,
        return_dict: bool = True,
    ):
        """
        Args:
            hidden_states (`Tensor` of shape `(batch size, height, width, channel)`):
                Input hidden_states.
            num_frames (`int`):
                The number of frames to be processed per batch. This is used to reshape the hidden states.
            encoder_hidden_states ( `Tensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            image_only_indicator (`Tensor` of shape `(batch size, num_frames)`, *optional*):
                A tensor indicating whether the input contains only images. 1 indicates that the input contains only
                images, 0 indicates that the input contains video frames.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformers.transformer_temporal.TransformerTemporalModelOutput`]
                instead of a plain tuple.

        Returns:
            [`~models.transformers.transformer_temporal.TransformerTemporalModelOutput`] or `tuple`:
                If `return_dict` is True, an
                [`~models.transformers.transformer_temporal.TransformerTemporalModelOutput`] is returned, otherwise a
                `tuple` where the first element is the sample tensor.
        """
        # 1. Input
        batch_frames, height, width, _ = ops.size()(hidden_states)
        num_frames = ops.size()(image_only_indicator, dim=-1)
        batch_size = batch_frames / num_frames

        time_context = encoder_hidden_states
        time_context_first_timestep = ops.dynamic_slice()(
            ops.reshape()(
                ops.unsqueeze(0)(time_context),
                [batch_size, num_frames, -1, ops.size()(time_context, dim=-1)],
            ),
            start_indices=[0, 0, 0, 0],
            end_indices=[None, 1, None, None],
        )
        time_context = ops.reshape()(
            ops.unsqueeze(0)(time_context_first_timestep),
            [
                batch_size,
                height * width,
                ops.size()(time_context, dim=-2),
                ops.size()(time_context, dim=-1),
            ],
        )
        time_context = ops.reshape()(
            time_context,
            [batch_size * height * width, -1, ops.size()(time_context, dim=-1)],
        )

        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = ops.size()(hidden_states, dim=-1)
        hidden_states = ops.reshape()(
            hidden_states, [batch_frames, height * width, inner_dim]
        )
        hidden_states = self.proj_in(hidden_states)

        num_frames_emb = ops.reshape()(
            ops.expand()(
                ops.unsqueeze(0)(ops.arange(0, num_frames._attrs["int_var"], 1)()),
                [batch_size, -1],
            ),
            [-1],
        )
        t_emb = self.time_proj(num_frames_emb)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = ops.cast()(t_emb, dtype=hidden_states.dtype())

        emb = self.time_pos_embed(t_emb)
        emb = ops.unsqueeze(1)(emb)

        # 2. Blocks
        for block, temporal_block in zip(
            self.transformer_blocks, self.temporal_transformer_blocks
        ):
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )

            hidden_states_mix = hidden_states
            hidden_states_mix = hidden_states_mix + emb

            hidden_states_mix = temporal_block(
                hidden_states_mix,
                num_frames=num_frames,
                encoder_hidden_states=time_context,
            )
            hidden_states = self.time_mixer(
                x_spatial=hidden_states,
                x_temporal=hidden_states_mix,
                image_only_indicator=image_only_indicator,
            )

        # 3. Output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = ops.reshape()(
            hidden_states, [batch_frames, height, width, inner_dim]
        )

        output = hidden_states + residual

        if not return_dict:
            return (output,)

        return TransformerTemporalModelOutput(sample=output)
