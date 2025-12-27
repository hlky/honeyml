from typing import Any, Dict, Optional, Tuple, Union

from dinoml.compiler import ops

from dinoml.frontend import nn, Tensor

from ..activations import get_activation
from ..attention import Attention, FeedForward
from ..attention_processor import AttentionProcessor
from ..embeddings import SiLU, TimestepEmbedding, Timesteps

from ..transformers.transformer_temporal import TransformerTemporalModel
from .unet_3d_blocks import (
    CrossAttnDownBlock3D,
    CrossAttnUpBlock3D,
    DownBlock3D,
    get_down_block,
    get_up_block,
    UNetMidBlock3DCrossAttn,
    UpBlock3D,
)
from .unet_3d_condition import UNet3DConditionOutput


class I2VGenXLTransformerTemporalEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        activation_fn: str = "geglu",
        upcast_attention: bool = False,
        ff_inner_dim: Optional[int] = None,
        dropout: int = 0.0,
        dtype: str = "float16",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=True, eps=1e-5, dtype=dtype)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=False,
            upcast_attention=upcast_attention,
            out_bias=True,
            dtype=dtype,
        )
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=False,
            inner_dim=ff_inner_dim,
            bias=True,
            dtype=dtype,
        )

    def forward(
        self,
        hidden_states: Tensor,
    ) -> Tensor:
        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(norm_hidden_states, encoder_hidden_states=None)
        hidden_states = attn_output + hidden_states
        if len(ops.size()(hidden_states)) == 4:
            hidden_states = ops.squeeze(1)(hidden_states)

        ff_output = self.ff(hidden_states)
        hidden_states = ff_output + hidden_states
        if len(ops.size()(hidden_states)) == 4:
            hidden_states = ops.squeeze(1)(hidden_states)

        return hidden_states


class I2VGenXLUNet(nn.Module):
    r"""
    I2VGenXL UNet. It is a conditional 3D UNet model that takes a noisy sample, conditional state, and a timestep and
    returns a sample-shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample.
        in_channels (`int`, *optional*, defaults to 4): The number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 4): The number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")`):
            The tuple of downsample blocks to use.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D")`):
            The tuple of upsample blocks to use.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2): The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32): The number of groups to use for the normalization.
            If `None`, normalization and activation layers is skipped in post-processing.
        cross_attention_dim (`int`, *optional*, defaults to 1280): The dimension of the cross attention features.
        attention_head_dim (`int`, *optional*, defaults to 64): Attention head dim.
        num_attention_heads (`int`, *optional*): The number of attention heads.
    """

    _supports_gradient_checkpointing = False

    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        down_block_types: Tuple[str, ...] = (
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D",
        ),
        up_block_types: Tuple[str, ...] = (
            "UpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        norm_num_groups: Optional[int] = 32,
        cross_attention_dim: int = 1024,
        attention_head_dim: Union[int, Tuple[int]] = 64,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
        dtype: str = "float16",
    ):
        super().__init__()
        self.dtype = dtype
        self.cross_attention_dim = cross_attention_dim

        # When we first integrated the UNet into the library, we didn't have `attention_head_dim`. As a consequence
        # of that, we used `num_attention_heads` for arguments that actually denote attention head dimension. This
        # is why we ignore `num_attention_heads` and calculate it from `attention_head_dims` below.
        # This is still an incorrect way of calculating `num_attention_heads` but we need to stick to it
        # without running proper depcrecation cycles for the {down,mid,up} blocks which are a
        # part of the public API.
        num_attention_heads = attention_head_dim

        # Check inputs
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(
            down_block_types
        ):
            raise ValueError(
                f"Must provide the same number of `num_attention_heads` as `down_block_types`. `num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}."
            )

        # input
        self.conv_in = nn.Conv2d(
            in_channels + in_channels,
            block_out_channels[0],
            kernel_size=3,
            padding=1,
            dtype=dtype,
        )

        self.transformer_in = TransformerTemporalModel(
            num_attention_heads=8,
            attention_head_dim=num_attention_heads,
            in_channels=block_out_channels[0],
            num_layers=1,
            norm_num_groups=norm_num_groups,
            dtype=dtype,
        )

        # image embedding
        self.image_latents_proj_in = nn.Sequential(
            nn.Conv2d(4, in_channels * 4, 3, padding=1, dtype=dtype),
            SiLU(),
            nn.Conv2d(
                in_channels * 4, in_channels * 4, 3, stride=1, padding=1, dtype=dtype
            ),
            SiLU(),
            nn.Conv2d(
                in_channels * 4, in_channels, 3, stride=1, padding=1, dtype=dtype
            ),
        )
        self.image_latents_temporal_encoder = I2VGenXLTransformerTemporalEncoder(
            dim=in_channels,
            num_attention_heads=2,
            ff_inner_dim=in_channels * 4,
            attention_head_dim=in_channels,
            activation_fn="gelu",
            dtype=dtype,
        )
        self.image_latents_context_embedding = nn.Sequential(
            nn.Conv2d(4, in_channels * 8, 3, padding=1, dtype=dtype),
            SiLU(),
            nn.AvgPool2d(32, 1, 0),
            nn.Conv2d(
                in_channels * 8, in_channels * 16, 3, stride=2, padding=1, dtype=dtype
            ),
            SiLU(),
            nn.Conv2d(
                in_channels * 16,
                cross_attention_dim,
                3,
                stride=2,
                padding=1,
                dtype=dtype,
            ),
        )

        # other embeddings -- time, context, fps, etc.
        time_embed_dim = block_out_channels[0] * 4
        self.time_proj = Timesteps(block_out_channels[0], True, 0, dtype=dtype)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim, time_embed_dim, act_fn="silu", dtype=dtype
        )
        self.context_embedding = nn.Sequential(
            nn.Linear(cross_attention_dim, time_embed_dim, dtype=dtype),
            SiLU(),
            nn.Linear(time_embed_dim, cross_attention_dim * in_channels, dtype=dtype),
        )
        self.fps_embedding = nn.Sequential(
            nn.Linear(timestep_input_dim, time_embed_dim, dtype=dtype),
            SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim, dtype=dtype),
        )

        # blocks
        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=1e-05,
                resnet_act_fn="silu",
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                num_attention_heads=num_attention_heads[i],
                downsample_padding=1,
                dual_cross_attention=False,
                dtype=dtype,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock3DCrossAttn(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            resnet_eps=1e-05,
            resnet_act_fn="silu",
            output_scale_factor=1,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads[-1],
            resnet_groups=norm_num_groups,
            dual_cross_attention=False,
            dtype=dtype,
        )

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[
                min(i + 1, len(block_out_channels) - 1)
            ]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=1e-05,
                resnet_act_fn="silu",
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                num_attention_heads=reversed_num_attention_heads[i],
                dual_cross_attention=False,
                resolution_idx=i,
                dtype=dtype,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0],
            num_groups=norm_num_groups,
            eps=1e-05,
            dtype=dtype,
        )
        self.conv_act = get_activation("silu")
        self.conv_out = nn.Conv2d(
            block_out_channels[0], out_channels, kernel_size=3, padding=1, dtype=dtype
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
        sample: Tensor,
        timestep: Tensor,
        fps: Tensor,
        image_latents: Tensor,
        image_embeddings: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        timestep_cond: Optional[Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[UNet3DConditionOutput, Tuple[Tensor]]:
        r"""
        The [`I2VGenXLUNet`] forward method.

        Args:
            sample (`Tensor`):
                The noisy input tensor with the following shape `(batch, num_frames, height, width, channel`.
            timestep (`Tensor`): The number of timesteps to denoise an input.
            fps (`Tensor`): Frames per second for the video being generated. Used as a "micro-condition".
            image_latents (`Tensor`): Image encodings from the VAE.
            image_embeddings (`Tensor`):
                Projection embeddings of the conditioning image computed with a vision encoder.
            encoder_hidden_states (`Tensor`):
                The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_3d_condition.UNet3DConditionOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.unets.unet_3d_condition.UNet3DConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unets.unet_3d_condition.UNet3DConditionOutput`] is returned,
                otherwise a `tuple` is returned where the first element is the sample tensor.
        """
        batch_size, num_frames, height, width, channels = ops.size()(sample)

        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        for dim in [height, width]:
            if dim._attrs["int_var"] % default_overall_up_factor != 0:
                # Forward upsample size to force interpolation output size.
                forward_upsample_size = True
                break

        # 1. time
        timesteps = timestep

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = ops.expand()(timesteps, [batch_size])
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = ops.cast()(t_emb, dtype=self.dtype)
        t_emb = self.time_embedding(t_emb, timestep_cond)

        # 2. FPS
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        fps = ops.expand()(fps, shape=[ops.size()(fps, dim=0)])
        fps_emb = self.fps_embedding(ops.cast()(self.time_proj(fps), dtype=self.dtype))

        # 3. time + FPS embeddings.
        emb = t_emb + fps_emb
        emb = ops.repeat_interleave(num_frames, 0)(emb)

        # 4. context embeddings.
        # The context embeddings consist of both text embeddings from the input prompt
        # AND the image embeddings from the input image. For images, both VAE encodings
        # and the CLIP image embeddings are incorporated.
        # So the final `context_embeddings` becomes the query for cross-attention.
        context_emb = ops.full()(
            [batch_size, 0, self.cross_attention_dim], fill_value=0.0
        )
        context_emb = ops.concatenate()([context_emb, encoder_hidden_states], dim=1)

        image_latents_for_context_embds = ops.dynamic_slice()(
            image_latents, start_indices=[0, 0, 0, 0], end_indices=[None, None, 1, None]
        )
        image_latents_context_embs = ops.reshape()(
            image_latents_for_context_embds,
            [
                ops.size()(image_latents_for_context_embds, dim=0)
                * ops.size()(image_latents_for_context_embds, dim=1),
                ops.size()(image_latents_for_context_embds, dim=2),
                ops.size()(image_latents_for_context_embds, dim=3),
                ops.size()(image_latents_for_context_embds, dim=4),
            ],
        )
        image_latents_context_embs = self.image_latents_context_embedding(
            image_latents_context_embs
        )

        _batch_size, _height, _width, _channels = ops.size()(image_latents_context_embs)
        image_latents_context_embs = ops.reshape()(
            image_latents_context_embs, [_batch_size, _height * _width, _channels]
        )
        context_emb = ops.concatenate()(
            [context_emb, image_latents_context_embs], dim=1
        )

        image_emb = self.context_embedding(image_embeddings)
        image_emb = ops.reshape()(
            image_emb, [-1, self.config.in_channels, self.config.cross_attention_dim]
        )
        context_emb = ops.concatenate()([context_emb, image_emb], dim=1)
        context_emb = ops.repeat_interleave(num_frames, 0)(context_emb)

        image_latents = ops.reshape()(
            image_latents,
            [
                ops.size()(image_latents, dim=0) * ops.size()(image_latents, dim=1),
                ops.size()(image_latents, dim=2),
                ops.size()(image_latents, dim=3),
                ops.size()(image_latents, dim=4),
            ],
        )
        image_latents = self.image_latents_proj_in(image_latents)
        image_latents = ops.reshape()(
            ops.reshape()(
                ops.unsqueeze(0)(image_latents),
                [batch_size, num_frames, height, width, channels],
            ),
            [batch_size * height * width, num_frames, channels],
        )
        image_latents = self.image_latents_temporal_encoder(image_latents)
        image_latents = ops.permute()(
            ops.reshape()(
                image_latents, [batch_size, height, width, num_frames, channels]
            ),
            [0, 3, 1, 2, 4],
        )

        # 5. pre-process
        sample = ops.concatenate()([sample, image_latents], dim=1)
        sample = ops.reshape()(
            sample, [batch_size * num_frames, height, width, channels]
        )
        sample = self.conv_in(sample)
        sample = self.transformer_in(
            sample,
            num_frames=num_frames,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
        )[0]

        # 6. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if (
                hasattr(downsample_block, "has_cross_attention")
                and downsample_block.has_cross_attention
            ):
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=context_emb,
                    num_frames=num_frames,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample, temb=emb, num_frames=num_frames
                )

            down_block_res_samples += res_samples

        # 7. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=context_emb,
                num_frames=num_frames,
                cross_attention_kwargs=cross_attention_kwargs,
            )
        # 8. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[
                : -len(upsample_block.resnets)
            ]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = ops.size()(down_block_res_samples[-1])[1:3]

            if (
                hasattr(upsample_block, "has_cross_attention")
                and upsample_block.has_cross_attention
            ):
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=context_emb,
                    upsample_size=upsample_size,
                    num_frames=num_frames,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                    num_frames=num_frames,
                )

        # 9. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)

        sample = self.conv_out(sample)

        # reshape to (batch, framerate, width, height, channel)
        sample = ops.reshape()(
            ops.unsqueeze(0)(sample), [-1, num_frames] + ops.size()(sample)[1:]
        )

        if not return_dict:
            return (sample,)

        return UNet3DConditionOutput(sample=sample)
