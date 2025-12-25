from typing import Any, Dict, Optional, Tuple, Union

from honey.compiler import ops

from honey.frontend import nn, Tensor

from ..attention_processor import AttentionProcessor
from ..embeddings import TimestepEmbedding, Timesteps

from ..transformers.transformer_temporal import TransformerTemporalModel
from .unet_2d_blocks import UNetMidBlock2DCrossAttn
from .unet_2d_condition import UNet2DConditionModel
from .unet_3d_blocks import (
    CrossAttnDownBlockMotion,
    CrossAttnUpBlockMotion,
    DownBlockMotion,
    get_down_block,
    get_up_block,
    UNetMidBlockCrossAttnMotion,
    UpBlockMotion,
)
from .unet_3d_condition import UNet3DConditionOutput


class MotionModules(nn.Module):
    def __init__(
        self,
        in_channels: int,
        layers_per_block: int = 2,
        num_attention_heads: int = 8,
        attention_bias: bool = False,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        norm_num_groups: int = 32,
        max_seq_length: int = 32,
        dtype: str = "float16",
    ):
        super().__init__()
        self.motion_modules = nn.ModuleList([])

        for i in range(layers_per_block):
            self.motion_modules.append(
                TransformerTemporalModel(
                    in_channels=in_channels,
                    norm_num_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=in_channels // num_attention_heads,
                    positional_embeddings="sinusoidal",
                    num_positional_embeddings=max_seq_length,
                    dtype=dtype,
                )
            )


class MotionAdapter(nn.Module):
    def __init__(
        self,
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        motion_layers_per_block: int = 2,
        motion_mid_block_layers_per_block: int = 1,
        motion_num_attention_heads: int = 8,
        motion_norm_num_groups: int = 32,
        motion_max_seq_length: int = 32,
        use_motion_mid_block: bool = True,
        conv_in_channels: Optional[int] = None,
        dtype: str = "float16",
    ):
        """Container to store AnimateDiff Motion Modules

        Args:
            block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each UNet block.
            motion_layers_per_block (`int`, *optional*, defaults to 2):
                The number of motion layers per UNet block.
            motion_mid_block_layers_per_block (`int`, *optional*, defaults to 1):
                The number of motion layers in the middle UNet block.
            motion_num_attention_heads (`int`, *optional*, defaults to 8):
                The number of heads to use in each attention layer of the motion module.
            motion_norm_num_groups (`int`, *optional*, defaults to 32):
                The number of groups to use in each group normalization layer of the motion module.
            motion_max_seq_length (`int`, *optional*, defaults to 32):
                The maximum sequence length to use in the motion module.
            use_motion_mid_block (`bool`, *optional*, defaults to True):
                Whether to use a motion module in the middle of the UNet.
        """

        super().__init__()
        down_blocks = []
        up_blocks = []

        if conv_in_channels:
            # input
            self.conv_in = nn.Conv2d(
                conv_in_channels,
                block_out_channels[0],
                kernel_size=3,
                padding=1,
                dtype=dtype,
            )
        else:
            self.conv_in = None

        for i, channel in enumerate(block_out_channels):
            output_channel = block_out_channels[i]
            down_blocks.append(
                MotionModules(
                    in_channels=output_channel,
                    norm_num_groups=motion_norm_num_groups,
                    cross_attention_dim=None,
                    activation_fn="geglu",
                    attention_bias=False,
                    num_attention_heads=motion_num_attention_heads,
                    max_seq_length=motion_max_seq_length,
                    layers_per_block=motion_layers_per_block,
                    dtype=dtype,
                )
            )

        if use_motion_mid_block:
            self.mid_block = MotionModules(
                in_channels=block_out_channels[-1],
                norm_num_groups=motion_norm_num_groups,
                cross_attention_dim=None,
                activation_fn="geglu",
                attention_bias=False,
                num_attention_heads=motion_num_attention_heads,
                layers_per_block=motion_mid_block_layers_per_block,
                max_seq_length=motion_max_seq_length,
                dtype=dtype,
            )
        else:
            self.mid_block = None

        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, channel in enumerate(reversed_block_out_channels):
            output_channel = reversed_block_out_channels[i]
            up_blocks.append(
                MotionModules(
                    in_channels=output_channel,
                    norm_num_groups=motion_norm_num_groups,
                    cross_attention_dim=None,
                    activation_fn="geglu",
                    attention_bias=False,
                    num_attention_heads=motion_num_attention_heads,
                    max_seq_length=motion_max_seq_length,
                    layers_per_block=motion_layers_per_block + 1,
                    dtype=dtype,
                )
            )

        self.down_blocks = nn.ModuleList(down_blocks)
        self.up_blocks = nn.ModuleList(up_blocks)

    def forward(self, sample):
        pass


class UNetMotionModel(nn.Module):
    r"""
    A modified conditional 2D UNet model that takes a noisy sample, conditional state, and a timestep and returns a
    sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        down_block_types: Tuple[str, ...] = (
            "CrossAttnDownBlockMotion",
            "CrossAttnDownBlockMotion",
            "CrossAttnDownBlockMotion",
            "DownBlockMotion",
        ),
        up_block_types: Tuple[str, ...] = (
            "UpBlockMotion",
            "CrossAttnUpBlockMotion",
            "CrossAttnUpBlockMotion",
            "CrossAttnUpBlockMotion",
        ),
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        reverse_transformer_layers_per_block: Optional[Tuple[Tuple[int]]] = None,
        use_linear_projection: bool = False,
        num_attention_heads: Union[int, Tuple[int, ...]] = 8,
        motion_max_seq_length: int = 32,
        motion_num_attention_heads: int = 8,
        use_motion_mid_block: int = True,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        projection_class_embeddings_input_dim: Optional[int] = None,
        time_cond_proj_dim: Optional[int] = None,
        dtype: str = "float16",
        **kwargs,
    ):
        super().__init__()

        self.sample_size = sample_size
        self.addition_embed_type = addition_embed_type
        self.encoder_hid_dim_type = encoder_hid_dim_type

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

        if isinstance(cross_attention_dim, list) and len(cross_attention_dim) != len(
            down_block_types
        ):
            raise ValueError(
                f"Must provide the same number of `cross_attention_dim` as `down_block_types`. `cross_attention_dim`: {cross_attention_dim}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(layers_per_block, int) and len(layers_per_block) != len(
            down_block_types
        ):
            raise ValueError(
                f"Must provide the same number of `layers_per_block` as `down_block_types`. `layers_per_block`: {layers_per_block}. `down_block_types`: {down_block_types}."
            )

        if (
            isinstance(transformer_layers_per_block, list)
            and reverse_transformer_layers_per_block is None
        ):
            for layer_number_per_block in transformer_layers_per_block:
                if isinstance(layer_number_per_block, list):
                    raise ValueError(
                        "Must provide 'reverse_transformer_layers_per_block` if using asymmetrical UNet."
                    )

        # input
        conv_in_kernel = 3
        conv_out_kernel = 3
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=conv_in_kernel,
            padding=conv_in_padding,
            dtype=dtype,
        )

        # time
        time_embed_dim = block_out_channels[0] * 4
        self.time_proj = Timesteps(
            block_out_channels[0],
            True,
            0,
            dtype=dtype,
        )
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
            cond_proj_dim=time_cond_proj_dim,
            dtype=dtype,
        )

        if encoder_hid_dim_type is None:
            self.encoder_hid_proj = None

        if addition_embed_type == "text_time":
            self.add_time_proj = Timesteps(
                addition_time_embed_dim,
                True,
                0,
                dtype=dtype,
            )
            self.add_embedding = TimestepEmbedding(
                projection_class_embeddings_input_dim, time_embed_dim, dtype=dtype
            )

        # class embedding
        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(
                down_block_types
            )

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim[i],
                num_attention_heads=num_attention_heads[i],
                downsample_padding=downsample_padding,
                use_linear_projection=use_linear_projection,
                dual_cross_attention=False,
                temporal_num_attention_heads=motion_num_attention_heads,
                temporal_max_seq_length=motion_max_seq_length,
                transformer_layers_per_block=transformer_layers_per_block[i],
                dtype=dtype,
            )
            self.down_blocks.append(down_block)

        # mid
        if use_motion_mid_block:
            self.mid_block = UNetMidBlockCrossAttnMotion(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                cross_attention_dim=cross_attention_dim[-1],
                num_attention_heads=num_attention_heads[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=False,
                use_linear_projection=use_linear_projection,
                temporal_num_attention_heads=motion_num_attention_heads,
                temporal_max_seq_length=motion_max_seq_length,
                transformer_layers_per_block=transformer_layers_per_block[-1],
                dtype=dtype,
            )

        else:
            self.mid_block = UNetMidBlock2DCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                cross_attention_dim=cross_attention_dim[-1],
                num_attention_heads=num_attention_heads[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=False,
                use_linear_projection=use_linear_projection,
                transformer_layers_per_block=transformer_layers_per_block[-1],
                dtype=dtype,
            )

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))
        reversed_transformer_layers_per_block = list(
            reversed(transformer_layers_per_block)
        )

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
                num_layers=reversed_layers_per_block[i] + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=reversed_cross_attention_dim[i],
                num_attention_heads=reversed_num_attention_heads[i],
                dual_cross_attention=False,
                resolution_idx=i,
                use_linear_projection=use_linear_projection,
                temporal_num_attention_heads=motion_num_attention_heads,
                temporal_max_seq_length=motion_max_seq_length,
                transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                dtype=dtype,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if norm_num_groups is not None:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0],
                num_groups=norm_num_groups,
                eps=norm_eps,
                dtype=dtype,
            )
            self.conv_act = ops.silu
        else:
            self.conv_norm_out = None
            self.conv_act = None

        conv_out_padding = (conv_out_kernel - 1) // 2
        self.conv_out = nn.Conv2d(
            block_out_channels[0],
            out_channels,
            kernel_size=conv_out_kernel,
            padding=conv_out_padding,
            dtype=dtype,
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
        encoder_hidden_states: Tensor,
        timestep_cond: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[Tensor]] = None,
        mid_block_additional_residual: Optional[Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet3DConditionOutput, Tuple[Tensor]]:
        r"""
        The [`UNetMotionModel`] forward method.

        Args:
            sample (`Tensor`):
                The noisy input tensor with the following shape `(batch, num_frames, height, width, channel`.
            timestep (`Tensor`): The number of timesteps to denoise an input.
            encoder_hidden_states (`Tensor`):
                The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
            timestep_cond: (`Tensor`, *optional*, defaults to `None`):
                Conditional embeddings for timestep. If provided, the embeddings will be summed with the samples passed
                through the `self.time_embedding` layer to obtain the timestep embeddings.
            attention_mask (`Tensor`, *optional*, defaults to `None`):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            down_block_additional_residuals: (`tuple` of `Tensor`, *optional*):
                A tuple of tensors that if specified are added to the residuals of down unet blocks.
            mid_block_additional_residual: (`Tensor`, *optional*):
                A tensor that if specified is added to the residual of the middle unet block.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_3d_condition.UNet3DConditionOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.unets.unet_3d_condition.UNet3DConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unets.unet_3d_condition.UNet3DConditionOutput`] is returned,
                otherwise a `tuple` is returned where the first element is the sample tensor.
        """
        batch, frames, height, width, channel = ops.size()(sample)
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

        # prepare attention_mask
        if attention_mask is not None and len(ops.size()(attention_mask)) == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (
                1 - ops.cast()(attention_mask, dtype=sample.dtype())
            ) * -10000.0
            attention_mask = ops.unsqueeze(1)(attention_mask)

        # 1. time
        timesteps = timestep

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        num_frames = ops.size()(sample, dim=1)
        timesteps = ops.expand()(timesteps, [ops.size()(sample, dim=0)])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = ops.cast()(t_emb, dtype=self.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        if self.addition_embed_type == "text_time":
            if "text_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                )

            text_embeds = added_cond_kwargs.get("text_embeds")
            if "time_ids" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                )
            time_ids = added_cond_kwargs.get("time_ids")
            time_embeds = self.add_time_proj(ops.flatten()(time_ids))
            time_embeds = ops.reshape()(
                time_embeds, [ops.size()(text_embeds, dim=0), -1]
            )

            add_embeds = ops.concatenate()([text_embeds, time_embeds], dim=-1)
            add_embeds = ops.cast()(add_embeds, emb.dtype())
            aug_emb = self.add_embedding(add_embeds)

        emb = emb if aug_emb is None else emb + aug_emb
        emb = ops.repeat_interleave(num_frames, 0)(emb)
        encoder_hidden_states = ops.repeat_interleave(num_frames, 0)(
            encoder_hidden_states
        )

        if (
            self.encoder_hid_proj is not None
            and self.encoder_hid_dim_type == "ip_image_proj"
        ):
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'ip_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                )
            image_embeds = added_cond_kwargs.get("image_embeds")
            image_embeds = self.encoder_hid_proj(image_embeds)
            image_embeds = [
                ops.repeat_interleave(num_frames, 0)(image_embed)
                for image_embed in image_embeds
            ]
            encoder_hidden_states = (encoder_hidden_states, image_embeds)

        # 2. pre-process
        sample = ops.reshape()(sample, [batch * num_frames, height, width, channel])
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if (
                hasattr(downsample_block, "has_cross_attention")
                and downsample_block.has_cross_attention
            ):
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    num_frames=num_frames,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample, temb=emb, num_frames=num_frames
                )

            down_block_res_samples += res_samples

        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = (
                    down_block_res_sample + down_block_additional_residual
                )
                new_down_block_res_samples += (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            # To support older versions of motion modules that don't have a mid_block
            if hasattr(self.mid_block, "motion_modules"):
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    num_frames=num_frames,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )

        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        # 5. up
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
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
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

        # 6. post-process
        if self.conv_norm_out:
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
