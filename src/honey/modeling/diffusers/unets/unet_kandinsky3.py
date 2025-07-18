from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

from honey.compiler import ops

from honey.frontend import IntVar, nn, Tensor

from ..attention_processor import Attention, AttentionProcessor
from ..embeddings import SiLU, TimestepEmbedding, Timesteps

from ..utils import BaseOutput


def get_shape(x):
    shape = [
        it.value() if not isinstance(it, IntVar) else it.symbolic_value()
        for it in x._attrs["shape"]
    ]
    return shape


@dataclass
class Kandinsky3UNetOutput(BaseOutput):
    sample: Tensor = None


class Kandinsky3EncoderProj(nn.Module):
    def __init__(
        self, encoder_hid_dim: int, cross_attention_dim: int, dtype: str = "float16"
    ):
        super().__init__()
        self.projection_linear = nn.Linear(
            encoder_hid_dim, cross_attention_dim, bias=False, dtype=dtype
        )
        self.projection_norm = nn.LayerNorm(cross_attention_dim, dtype=dtype)

    def forward(self, x):
        x = self.projection_linear(x)
        x = self.projection_norm(x)
        return x


class Kandinsky3UNet(nn.Module):

    def __init__(
        self,
        in_channels: int = 4,
        time_embedding_dim: int = 1536,
        groups: int = 32,
        attention_head_dim: int = 64,
        layers_per_block: Union[int, Tuple[int]] = 3,
        block_out_channels: Tuple[int] = (384, 768, 1536, 3072),
        cross_attention_dim: Union[int, Tuple[int]] = 4096,
        encoder_hid_dim: int = 4096,
        dtype: str = "float16",
        **kwargs,
    ):
        super().__init__()

        # TOOD(Yiyi): Give better name and put into config for the following 4 parameters
        expansion_ratio = 4
        compression_ratio = 2
        add_cross_attention = (False, True, True, True)
        add_self_attention = (False, True, True, True)

        out_channels = in_channels
        init_channels = block_out_channels[0] // 2
        self.time_proj = Timesteps(
            init_channels, flip_sin_to_cos=False, downscale_freq_shift=1, dtype=dtype
        )

        self.time_embedding = TimestepEmbedding(
            init_channels, time_embedding_dim, dtype=dtype
        )

        self.add_time_condition = Kandinsky3AttentionPooling(
            time_embedding_dim, cross_attention_dim, attention_head_dim, dtype=dtype
        )

        self.conv_in = nn.Conv2d(
            in_channels, init_channels, kernel_size=3, padding=1, dtype=dtype
        )

        self.encoder_hid_proj = Kandinsky3EncoderProj(
            encoder_hid_dim, cross_attention_dim, dtype=dtype
        )

        hidden_dims = [init_channels] + list(block_out_channels)
        in_out_dims = list(zip(hidden_dims[:-1], hidden_dims[1:]))
        text_dims = [
            cross_attention_dim if is_exist else None
            for is_exist in add_cross_attention
        ]
        num_blocks = len(block_out_channels) * [layers_per_block]
        layer_params = [num_blocks, text_dims, add_self_attention]
        rev_layer_params = map(reversed, layer_params)

        cat_dims = []
        self.num_levels = len(in_out_dims)
        self.down_blocks = nn.ModuleList([])
        for level, (
            (in_dim, out_dim),
            res_block_num,
            text_dim,
            self_attention,
        ) in enumerate(zip(in_out_dims, *layer_params)):
            down_sample = level != (self.num_levels - 1)
            cat_dims.append(out_dim if level != (self.num_levels - 1) else 0)
            self.down_blocks.append(
                Kandinsky3DownSampleBlock(
                    in_dim,
                    out_dim,
                    time_embedding_dim,
                    text_dim,
                    res_block_num,
                    groups,
                    attention_head_dim,
                    expansion_ratio,
                    compression_ratio,
                    down_sample,
                    self_attention,
                    dtype=dtype,
                )
            )

        self.up_blocks = nn.ModuleList([])
        for level, (
            (out_dim, in_dim),
            res_block_num,
            text_dim,
            self_attention,
        ) in enumerate(zip(reversed(in_out_dims), *rev_layer_params)):
            up_sample = level != 0
            self.up_blocks.append(
                Kandinsky3UpSampleBlock(
                    in_dim,
                    cat_dims.pop(),
                    out_dim,
                    time_embedding_dim,
                    text_dim,
                    res_block_num,
                    groups,
                    attention_head_dim,
                    expansion_ratio,
                    compression_ratio,
                    up_sample,
                    self_attention,
                    dtype=dtype,
                )
            )

        self.conv_norm_out = nn.GroupNorm(groups, init_channels)
        self.conv_act_out = ops.silu
        self.conv_out = nn.Conv2d(
            init_channels, out_channels, kernel_size=3, padding=1, dtype=dtype
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
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        return_dict: bool = True,
    ):
        if encoder_attention_mask is not None:
            encoder_attention_mask = (
                1 - ops.cast()(encoder_attention_mask, dtype=hidden_states.dtype())
            ) * -10000.0
            encoder_attention_mask = ops.unsqueeze(1)(encoder_attention_mask)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = ops.expand()(timestep, [ops.size()(sample, dim=0)])
        time_embed_input = ops.cast()(self.time_proj(timestep), dtype=sample.dtype())
        time_embed = self.time_embedding(time_embed_input)

        encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)

        if encoder_hidden_states is not None:
            time_embed = self.add_time_condition(
                time_embed, encoder_hidden_states, encoder_attention_mask
            )

        hidden_states = []
        sample = self.conv_in(sample)
        for level, down_sample in enumerate(self.down_blocks):
            sample = down_sample(
                sample, time_embed, encoder_hidden_states, encoder_attention_mask
            )
            if level != self.num_levels - 1:
                hidden_states.append(sample)

        for level, up_sample in enumerate(self.up_blocks):
            if level != 0:
                temp_hidden_states = hidden_states.pop()
                temp_hidden_states._attrs["shape"] = sample._attrs["shape"]
                sample = ops.concatenate()([sample, temp_hidden_states], dim=-1)
            sample = up_sample(
                sample, time_embed, encoder_hidden_states, encoder_attention_mask
            )

        sample = self.conv_norm_out(sample)
        sample = self.conv_act_out(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)
        return Kandinsky3UNetOutput(sample=sample)


class Kandinsky3UpSampleBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        cat_dim,
        out_channels,
        time_embed_dim,
        context_dim=None,
        num_blocks=3,
        groups=32,
        head_dim=64,
        expansion_ratio=4,
        compression_ratio=2,
        up_sample=True,
        self_attention=True,
        dtype: str = "float16",
    ):
        super().__init__()
        up_resolutions = [[None, True if up_sample else None, None, None]] + [
            [None] * 4
        ] * (num_blocks - 1)
        hidden_channels = (
            [(in_channels + cat_dim, in_channels)]
            + [(in_channels, in_channels)] * (num_blocks - 2)
            + [(in_channels, out_channels)]
        )
        attentions = []
        resnets_in = []
        resnets_out = []

        self.self_attention = self_attention
        self.context_dim = context_dim

        if self_attention:
            attentions.append(
                Kandinsky3AttentionBlock(
                    out_channels,
                    time_embed_dim,
                    None,
                    groups,
                    head_dim,
                    expansion_ratio,
                    dtype=dtype,
                )
            )
        else:
            attentions.append(nn.Identity())

        for (in_channel, out_channel), up_resolution in zip(
            hidden_channels, up_resolutions
        ):
            resnets_in.append(
                Kandinsky3ResNetBlock(
                    in_channel,
                    in_channel,
                    time_embed_dim,
                    groups,
                    compression_ratio,
                    up_resolution,
                    dtype=dtype,
                )
            )

            if context_dim is not None:
                attentions.append(
                    Kandinsky3AttentionBlock(
                        in_channel,
                        time_embed_dim,
                        context_dim,
                        groups,
                        head_dim,
                        expansion_ratio,
                        dtype=dtype,
                    )
                )
            else:
                attentions.append(nn.Identity())

            resnets_out.append(
                Kandinsky3ResNetBlock(
                    in_channel,
                    out_channel,
                    time_embed_dim,
                    groups,
                    compression_ratio,
                    dtype=dtype,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets_in = nn.ModuleList(resnets_in)
        self.resnets_out = nn.ModuleList(resnets_out)

    def forward(self, x, time_embed, context=None, context_mask=None, image_mask=None):
        for attention, resnet_in, resnet_out in zip(
            self.attentions[1:], self.resnets_in, self.resnets_out
        ):
            x = resnet_in(x, time_embed)
            if self.context_dim is not None:
                x = attention(x, time_embed, context, context_mask, image_mask)
            x = resnet_out(x, time_embed)

        if self.self_attention:
            x = self.attentions[0](x, time_embed, image_mask=image_mask)
        return x


class Kandinsky3DownSampleBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        time_embed_dim,
        context_dim=None,
        num_blocks=3,
        groups=32,
        head_dim=64,
        expansion_ratio=4,
        compression_ratio=2,
        down_sample=True,
        self_attention=True,
        dtype: str = "float16",
    ):
        super().__init__()
        attentions = []
        resnets_in = []
        resnets_out = []

        self.self_attention = self_attention
        self.context_dim = context_dim

        if self_attention:
            attentions.append(
                Kandinsky3AttentionBlock(
                    in_channels,
                    time_embed_dim,
                    None,
                    groups,
                    head_dim,
                    expansion_ratio,
                    dtype=dtype,
                )
            )
        else:
            attentions.append(nn.Identity())

        up_resolutions = [[None] * 4] * (num_blocks - 1) + [
            [None, None, False if down_sample else None, None]
        ]
        hidden_channels = [(in_channels, out_channels)] + [
            (out_channels, out_channels)
        ] * (num_blocks - 1)
        for (in_channel, out_channel), up_resolution in zip(
            hidden_channels, up_resolutions
        ):
            resnets_in.append(
                Kandinsky3ResNetBlock(
                    in_channel,
                    out_channel,
                    time_embed_dim,
                    groups,
                    compression_ratio,
                    dtype=dtype,
                )
            )

            if context_dim is not None:
                attentions.append(
                    Kandinsky3AttentionBlock(
                        out_channel,
                        time_embed_dim,
                        context_dim,
                        groups,
                        head_dim,
                        expansion_ratio,
                        dtype=dtype,
                    )
                )
            else:
                attentions.append(nn.Identity())

            resnets_out.append(
                Kandinsky3ResNetBlock(
                    out_channel,
                    out_channel,
                    time_embed_dim,
                    groups,
                    compression_ratio,
                    up_resolution,
                    dtype=dtype,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets_in = nn.ModuleList(resnets_in)
        self.resnets_out = nn.ModuleList(resnets_out)

    def forward(self, x, time_embed, context=None, context_mask=None, image_mask=None):
        if self.self_attention:
            x = self.attentions[0](x, time_embed, image_mask=image_mask)

        for attention, resnet_in, resnet_out in zip(
            self.attentions[1:], self.resnets_in, self.resnets_out
        ):
            x = resnet_in(x, time_embed)
            if self.context_dim is not None:
                x = attention(x, time_embed, context, context_mask, image_mask)
            x = resnet_out(x, time_embed)
        return x


class Kandinsky3ConditionalGroupNorm(nn.Module):
    def __init__(
        self,
        groups: int,
        normalized_shape: int,
        context_dim: int,
        dtype: str = "float16",
    ):
        super().__init__()
        self.norm = nn.GroupNorm(groups, normalized_shape, affine=False, dtype=dtype)
        self.context_mlp = nn.Sequential(
            SiLU(), nn.Linear(context_dim, 2 * normalized_shape, dtype=dtype)
        )

    def forward(self, x, context):
        context = self.context_mlp(context)

        for _ in range(len(ops.size()(x)[1:3])):
            context = ops.unsqueeze(1)(context)

        scale, shift = ops.chunk()(context, 2, dim=-1)
        x = self.norm(x) * (scale + 1.0) + shift
        return x


class Kandinsky3Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        time_embed_dim,
        kernel_size=3,
        norm_groups=32,
        up_resolution=None,
        dtype: str = "float16",
    ):
        super().__init__()
        self.group_norm = Kandinsky3ConditionalGroupNorm(
            norm_groups, in_channels, time_embed_dim, dtype=dtype
        )
        self.activation = SiLU()
        if up_resolution is not None and up_resolution:
            self.up_sample = nn.ConvTranspose2dBias(
                in_channels, in_channels, kernel_size=2, stride=2, dtype=dtype
            )
        else:
            self.up_sample = nn.Identity()

        padding = int(kernel_size > 1)
        self.projection = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dtype=dtype,
        )

        if up_resolution is not None and not up_resolution:
            self.down_sample = nn.Conv2d(
                out_channels, out_channels, kernel_size=2, stride=2, dtype=dtype
            )
        else:
            self.down_sample = nn.Identity()

    def forward(self, x, time_embed):
        x = self.group_norm(x, time_embed)
        x = self.activation(x)
        x = self.up_sample(x)
        x = self.projection(x)
        x = self.down_sample(x)
        return x


class Kandinsky3ResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        time_embed_dim,
        norm_groups=32,
        compression_ratio=2,
        up_resolutions=4 * [None],
        dtype: str = "float16",
    ):
        super().__init__()
        kernel_sizes = [1, 3, 3, 1]
        hidden_channel = max(in_channels, out_channels) // compression_ratio
        hidden_channels = (
            [(in_channels, hidden_channel)]
            + [(hidden_channel, hidden_channel)] * 2
            + [(hidden_channel, out_channels)]
        )
        self.resnet_blocks = nn.ModuleList(
            [
                Kandinsky3Block(
                    in_channel,
                    out_channel,
                    time_embed_dim,
                    kernel_size,
                    norm_groups,
                    up_resolution,
                    dtype=dtype,
                )
                for (in_channel, out_channel), kernel_size, up_resolution in zip(
                    hidden_channels, kernel_sizes, up_resolutions
                )
            ]
        )
        self.shortcut_up_sample = (
            nn.ConvTranspose2dBias(
                in_channels, in_channels, kernel_size=2, stride=2, dtype=dtype
            )
            if True in up_resolutions
            else nn.Identity()
        )
        self.shortcut_projection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, dtype=dtype)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.shortcut_down_sample = (
            nn.Conv2d(
                out_channels, out_channels, kernel_size=2, stride=2, dtype=dtype
            )
            if False in up_resolutions
            else nn.Identity()
        )

    def forward(self, x: Tensor, time_embed: Tensor):
        out = x
        for resnet_block in self.resnet_blocks:
            out = resnet_block(out, time_embed)

        x = self.shortcut_up_sample(x)
        x = self.shortcut_projection(x)
        x = self.shortcut_down_sample(x)

        out._attrs["shape"] = x._attrs["shape"]
        x = x + out
        return x


class Kandinsky3AttentionPooling(nn.Module):
    def __init__(
        self,
        num_channels: int,
        context_dim: int,
        head_dim: int = 64,
        dtype: str = "float16",
    ):
        super().__init__()
        self.attention = Attention(
            context_dim,
            context_dim,
            dim_head=head_dim,
            out_dim=num_channels,
            out_bias=False,
            dtype=dtype,
        )

    def forward(
        self, x: Tensor, context: Tensor, context_mask: Optional[Tensor] = None
    ):
        if context_mask is not None:
            context_mask = ops.cast()(context_mask, dtype=context.dtype())
        context = self.attention(
            ops.reduce_mean(dim=1, keepdim=True)(context), context, context_mask
        )
        return x + ops.squeeze(1)(context)


class Kandinsky3AttentionBlock(nn.Module):
    def __init__(
        self,
        num_channels,
        time_embed_dim,
        context_dim=None,
        norm_groups=32,
        head_dim=64,
        expansion_ratio=4,
        dtype: str = "float16",
    ):
        super().__init__()
        self.in_norm = Kandinsky3ConditionalGroupNorm(
            norm_groups, num_channels, time_embed_dim, dtype=dtype
        )
        self.attention = Attention(
            num_channels,
            context_dim or num_channels,
            dim_head=head_dim,
            out_dim=num_channels,
            out_bias=False,
            dtype=dtype,
        )

        hidden_channels = expansion_ratio * num_channels
        self.out_norm = Kandinsky3ConditionalGroupNorm(
            norm_groups, num_channels, time_embed_dim, dtype=dtype
        )
        self.feed_forward = nn.Sequential(
            nn.Conv2d(num_channels, hidden_channels, kernel_size=1, dtype=dtype, bias=False),
            SiLU(),
            nn.Conv2d(hidden_channels, num_channels, kernel_size=1, dtype=dtype, bias=False),
        )

    def forward(
        self,
        x: Tensor,
        time_embed: Tensor,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        image_mask: Optional[Tensor] = None,
    ):
        height, width = ops.size()(x)[1:3]
        out = self.in_norm(x, time_embed)
        out = ops.reshape()(out, [ops.size()(x, dim=0), height * width, -1])
        context = context if context is not None else out
        if context_mask is not None:
            context_mask = ops.cast()(context_mask, dtype=context.dtype())

        out = self.attention(out, context, context_mask)
        out = ops.reshape()(
            ops.unsqueeze(-1)(out), [ops.size()(out, dim=0), height, width, -1]
        )
        x = x + out

        out = self.out_norm(x, time_embed)
        out = self.feed_forward(out)
        out._attrs["shape"] = x._attrs["shape"]
        x = x + out
        return x
