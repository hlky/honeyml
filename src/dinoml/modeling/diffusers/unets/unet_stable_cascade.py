import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

from dinoml.compiler import ops

from dinoml.frontend import nn, Tensor

from ..attention_processor import Attention
from ..embeddings import GELU, SiLU
from ..normalization import GlobalResponseNorm

from ..utils import BaseOutput


class SDCascadeLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        return super().forward(x)


class SDCascadeTimestepBlock(nn.Module):
    def __init__(self, c, c_timestep, conds=[], dtype: str = "float16"):
        super().__init__()

        self.mapper = nn.Linear(c_timestep, c * 2, dtype=dtype)
        self.conds = conds
        for cname in conds:
            setattr(self, f"mapper_{cname}", nn.Linear(c_timestep, c * 2, dtype=dtype))

    def forward(self, x, t):
        t = ops.chunk()(t, len(self.conds) + 1, dim=1)
        a, b = ops.chunk()(
            ops.unsqueeze(1)(ops.unsqueeze(1)(self.mapper(t[0]))), 2, dim=-1
        )
        for i, c in enumerate(self.conds):
            ac, bc = ops.chunk()(
                ops.unsqueeze(1)(
                    ops.unsqueeze(1)(getattr(self, f"mapper_{c}")(t[i + 1]))
                ),
                2,
                dim=-1,
            )
            a, b = a + ac, b + bc
        return x * (1 + a) + b


class SDCascadeResBlock(nn.Module):
    def __init__(self, c, c_skip=0, kernel_size=3, dropout=0.0, dtype: str = "float16"):
        super().__init__()
        self.depthwise = nn.Conv2d(
            c,
            c,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=c,
            dtype=dtype,
        )
        self.norm = SDCascadeLayerNorm(
            c, elementwise_affine=False, eps=1e-6, dtype=dtype
        )
        self.channelwise = nn.Sequential(
            nn.Linear(c + c_skip, c * 4, dtype=dtype),
            GELU(),
            GlobalResponseNorm(c * 4, dtype=dtype),
            nn.Dropout(dropout),
            nn.Linear(c * 4, c, dtype=dtype),
        )

    def forward(self, x, x_skip=None):
        x_res = x
        x = self.norm(self.depthwise(x))
        if x_skip is not None:
            x = ops.concatenate()([x, x_skip], dim=-1)
        x = self.channelwise(x)
        return x + x_res


class SDCascadeAttnBlock(nn.Module):
    def __init__(
        self, c, c_cond, nhead, self_attn=True, dropout=0.0, dtype: str = "float16"
    ):
        super().__init__()

        self.self_attn = self_attn
        self.norm = SDCascadeLayerNorm(
            c, elementwise_affine=False, eps=1e-6, dtype=dtype
        )
        self.attention = Attention(
            query_dim=c,
            heads=nhead,
            dim_head=c // nhead,
            dropout=dropout,
            bias=True,
            dtype=dtype,
        )
        self.kv_mapper = nn.Sequential(SiLU(), nn.Linear(c_cond, c))

    def forward(self, x: Tensor, kv: Tensor):
        kv = self.kv_mapper(kv)
        norm_x = self.norm(x)
        if self.self_attn:
            batch_size, _, _, channel = ops.size()(x)
            kv_0 = ops.permute021()(ops.reshape()(norm_x, [batch_size, channel, -1]))
            kv_0._attrs["shape"][0] = kv._attrs["shape"][0]
            kv = ops.concatenate()(
                [
                    kv_0,
                    kv,
                ],
                dim=1,
            )
        x = x + self.attention(norm_x, encoder_hidden_states=kv)
        return x


class UpDownBlock2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, mode, enabled=True, dtype: str = "float16"
    ):
        super().__init__()
        if mode not in ["up", "down"]:
            raise ValueError(f"{mode} not supported")
        interpolation = (
            ops.upsampling2d(
                scale_factor=2 if mode == "up" else 0.5,
                mode="bilinear",
                align_corners=True,
            )
            if enabled
            else nn.Identity()
        )
        mapping = nn.Conv2d(in_channels, out_channels, kernel_size=1, dtype=dtype)
        self.blocks = nn.ModuleList(
            [interpolation, mapping] if mode == "up" else [mapping, interpolation]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


@dataclass
class StableCascadeUNetOutput(BaseOutput):
    sample: Tensor = None


class StableCascadeUNet(nn.Module):
    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        timestep_ratio_embedding_dim: int = 64,
        patch_size: int = 1,
        conditioning_dim: int = 2048,
        block_out_channels: Tuple[int] = (2048, 2048),
        num_attention_heads: Tuple[int] = (32, 32),
        down_num_layers_per_block: Tuple[int] = (8, 24),
        up_num_layers_per_block: Tuple[int] = (24, 8),
        down_blocks_repeat_mappers: Optional[Tuple[int]] = (
            1,
            1,
        ),
        up_blocks_repeat_mappers: Optional[Tuple[int]] = (1, 1),
        block_types_per_layer: Tuple[Tuple[str]] = (
            ("SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"),
            ("SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"),
        ),
        clip_text_in_channels: Optional[int] = None,
        clip_text_pooled_in_channels=1280,
        clip_image_in_channels: Optional[int] = None,
        clip_seq=4,
        effnet_in_channels: Optional[int] = None,
        pixel_mapper_in_channels: Optional[int] = None,
        kernel_size=3,
        dropout: Union[float, Tuple[float]] = (0.1, 0.1),
        self_attn: Union[bool, Tuple[bool]] = True,
        timestep_conditioning_type: Tuple[str] = ("sca", "crp"),
        switch_level: Optional[Tuple[bool]] = None,
        dtype: str = "float16",
    ):
        """

        Parameters:
            in_channels (`int`, defaults to 16):
                Number of channels in the input sample.
            out_channels (`int`, defaults to 16):
                Number of channels in the output sample.
            timestep_ratio_embedding_dim (`int`, defaults to 64):
                Dimension of the projected time embedding.
            patch_size (`int`, defaults to 1):
                Patch size to use for pixel unshuffling layer
            conditioning_dim (`int`, defaults to 2048):
                Dimension of the image and text conditional embedding.
            block_out_channels (Tuple[int], defaults to (2048, 2048)):
                Tuple of output channels for each block.
            num_attention_heads (Tuple[int], defaults to (32, 32)):
                Number of attention heads in each attention block. Set to -1 to if block types in a layer do not have
                attention.
            down_num_layers_per_block (Tuple[int], defaults to [8, 24]):
                Number of layers in each down block.
            up_num_layers_per_block (Tuple[int], defaults to [24, 8]):
                Number of layers in each up block.
            down_blocks_repeat_mappers (Tuple[int], optional, defaults to [1, 1]):
                Number of 1x1 Convolutional layers to repeat in each down block.
            up_blocks_repeat_mappers (Tuple[int], optional, defaults to [1, 1]):
                Number of 1x1 Convolutional layers to repeat in each up block.
            block_types_per_layer (Tuple[Tuple[str]], optional,
                defaults to (
                    ("SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"), ("SDCascadeResBlock",
                    "SDCascadeTimestepBlock", "SDCascadeAttnBlock")
                ): Block types used in each layer of the up/down blocks.
            clip_text_in_channels (`int`, *optional*, defaults to `None`):
                Number of input channels for CLIP based text conditioning.
            clip_text_pooled_in_channels (`int`, *optional*, defaults to 1280):
                Number of input channels for pooled CLIP text embeddings.
            clip_image_in_channels (`int`, *optional*):
                Number of input channels for CLIP based image conditioning.
            clip_seq (`int`, *optional*, defaults to 4):
            effnet_in_channels (`int`, *optional*, defaults to `None`):
                Number of input channels for effnet conditioning.
            pixel_mapper_in_channels (`int`, defaults to `None`):
                Number of input channels for pixel mapper conditioning.
            kernel_size (`int`, *optional*, defaults to 3):
                Kernel size to use in the block convolutional layers.
            dropout (Tuple[float], *optional*, defaults to (0.1, 0.1)):
                Dropout to use per block.
            self_attn (Union[bool, Tuple[bool]]):
                Tuple of booleans that determine whether to use self attention in a block or not.
            timestep_conditioning_type (Tuple[str], defaults to ("sca", "crp")):
                Timestep conditioning type.
            switch_level (Optional[Tuple[bool]], *optional*, defaults to `None`):
                Tuple that indicates whether upsampling or downsampling should be applied in a block
        """

        super().__init__()
        self.timestep_ratio_embedding_dim = timestep_ratio_embedding_dim
        self.dtype = dtype
        self.clip_seq = clip_seq
        self.timestep_conditioning_type = timestep_conditioning_type

        if len(block_out_channels) != len(down_num_layers_per_block):
            raise ValueError(
                f"Number of elements in `down_num_layers_per_block` must match the length of `block_out_channels`: {len(block_out_channels)}"
            )

        elif len(block_out_channels) != len(up_num_layers_per_block):
            raise ValueError(
                f"Number of elements in `up_num_layers_per_block` must match the length of `block_out_channels`: {len(block_out_channels)}"
            )

        elif len(block_out_channels) != len(down_blocks_repeat_mappers):
            raise ValueError(
                f"Number of elements in `down_blocks_repeat_mappers` must match the length of `block_out_channels`: {len(block_out_channels)}"
            )

        elif len(block_out_channels) != len(up_blocks_repeat_mappers):
            raise ValueError(
                f"Number of elements in `up_blocks_repeat_mappers` must match the length of `block_out_channels`: {len(block_out_channels)}"
            )

        elif len(block_out_channels) != len(block_types_per_layer):
            raise ValueError(
                f"Number of elements in `block_types_per_layer` must match the length of `block_out_channels`: {len(block_out_channels)}"
            )

        if isinstance(dropout, float):
            dropout = (dropout,) * len(block_out_channels)
        if isinstance(self_attn, bool):
            self_attn = (self_attn,) * len(block_out_channels)

        # CONDITIONING
        if effnet_in_channels is not None:
            self.effnet_mapper = nn.Sequential(
                nn.Conv2d(
                    effnet_in_channels,
                    block_out_channels[0] * 4,
                    kernel_size=1,
                    dtype=dtype,
                ),
                GELU(),
                nn.Conv2d(
                    block_out_channels[0] * 4,
                    block_out_channels[0],
                    kernel_size=1,
                    dtype=dtype,
                ),
                SDCascadeLayerNorm(
                    block_out_channels[0],
                    elementwise_affine=False,
                    eps=1e-6,
                    dtype=dtype,
                ),
            )
        if pixel_mapper_in_channels is not None:
            self.pixels_mapper = nn.Sequential(
                nn.Conv2d(
                    pixel_mapper_in_channels,
                    block_out_channels[0] * 4,
                    kernel_size=1,
                    dtype=dtype,
                ),
                GELU(),
                nn.Conv2d(
                    block_out_channels[0] * 4,
                    block_out_channels[0],
                    kernel_size=1,
                    dtype=dtype,
                ),
                SDCascadeLayerNorm(
                    block_out_channels[0],
                    elementwise_affine=False,
                    eps=1e-6,
                    dtype=dtype,
                ),
            )

        self.clip_txt_pooled_mapper = nn.Linear(
            clip_text_pooled_in_channels, conditioning_dim * clip_seq, dtype=dtype
        )
        if clip_text_in_channels is not None:
            self.clip_txt_mapper = nn.Linear(
                clip_text_in_channels, conditioning_dim, dtype=dtype
            )
        if clip_image_in_channels is not None:
            self.clip_img_mapper = nn.Linear(
                clip_image_in_channels, conditioning_dim * clip_seq, dtype=dtype
            )
        self.clip_norm = nn.LayerNorm(
            conditioning_dim, elementwise_affine=False, eps=1e-6, dtype=dtype
        )

        self.embedding = nn.Sequential(
            nn.PixelUnshuffle(patch_size),
            nn.Conv2d(
                in_channels * (patch_size**2),
                block_out_channels[0],
                kernel_size=1,
                dtype=dtype,
            ),
            SDCascadeLayerNorm(
                block_out_channels[0], elementwise_affine=False, eps=1e-6, dtype=dtype
            ),
        )

        def get_block(
            block_type,
            in_channels,
            nhead,
            c_skip=0,
            dropout=0,
            self_attn=True,
            dtype: str = "float16",
        ):
            if block_type == "SDCascadeResBlock":
                return SDCascadeResBlock(
                    in_channels,
                    c_skip,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    dtype=dtype,
                )
            elif block_type == "SDCascadeAttnBlock":
                return SDCascadeAttnBlock(
                    in_channels,
                    conditioning_dim,
                    nhead,
                    self_attn=self_attn,
                    dropout=dropout,
                    dtype=dtype,
                )
            elif block_type == "SDCascadeTimestepBlock":
                return SDCascadeTimestepBlock(
                    in_channels,
                    timestep_ratio_embedding_dim,
                    conds=timestep_conditioning_type,
                    dtype=dtype,
                )
            else:
                raise ValueError(f"Block type {block_type} not supported")

        # BLOCKS
        # -- down blocks
        self.down_blocks = nn.ModuleList()
        self.down_downscalers = nn.ModuleList()
        self.down_repeat_mappers = nn.ModuleList()
        for i in range(len(block_out_channels)):
            if i > 0:
                self.down_downscalers.append(
                    nn.Sequential(
                        SDCascadeLayerNorm(
                            block_out_channels[i - 1],
                            elementwise_affine=False,
                            eps=1e-6,
                            dtype=dtype,
                        ),
                        (
                            UpDownBlock2d(
                                block_out_channels[i - 1],
                                block_out_channels[i],
                                mode="down",
                                enabled=switch_level[i - 1],
                                dtype=dtype,
                            )
                            if switch_level is not None
                            else nn.Conv2d(
                                block_out_channels[i - 1],
                                block_out_channels[i],
                                kernel_size=2,
                                stride=2,
                                dtype=dtype,
                            )
                        ),
                    )
                )
            else:
                self.down_downscalers.append(nn.Identity())

            down_block = nn.ModuleList()
            for _ in range(down_num_layers_per_block[i]):
                for block_type in block_types_per_layer[i]:
                    block = get_block(
                        block_type,
                        block_out_channels[i],
                        num_attention_heads[i],
                        dropout=dropout[i],
                        self_attn=self_attn[i],
                        dtype=dtype,
                    )
                    down_block.append(block)
            self.down_blocks.append(down_block)

            if down_blocks_repeat_mappers is not None:
                block_repeat_mappers = nn.ModuleList()
                for _ in range(down_blocks_repeat_mappers[i] - 1):
                    block_repeat_mappers.append(
                        nn.Conv2d(
                            block_out_channels[i],
                            block_out_channels[i],
                            kernel_size=1,
                            dtype=dtype,
                        )
                    )
                self.down_repeat_mappers.append(block_repeat_mappers)

        # -- up blocks
        self.up_blocks = nn.ModuleList()
        self.up_upscalers = nn.ModuleList()
        self.up_repeat_mappers = nn.ModuleList()
        for i in reversed(range(len(block_out_channels))):
            if i > 0:
                self.up_upscalers.append(
                    nn.Sequential(
                        SDCascadeLayerNorm(
                            block_out_channels[i],
                            elementwise_affine=False,
                            eps=1e-6,
                            dtype=dtype,
                        ),
                        (
                            UpDownBlock2d(
                                block_out_channels[i],
                                block_out_channels[i - 1],
                                mode="up",
                                enabled=switch_level[i - 1],
                                dtype=dtype,
                            )
                            if switch_level is not None
                            else nn.ConvTranspose2dBias(
                                block_out_channels[i],
                                block_out_channels[i - 1],
                                kernel_size=2,
                                stride=2,
                                dtype=dtype,
                            )
                        ),
                    )
                )
            else:
                self.up_upscalers.append(nn.Identity())

            up_block = nn.ModuleList()
            for j in range(up_num_layers_per_block[::-1][i]):
                for k, block_type in enumerate(block_types_per_layer[i]):
                    c_skip = (
                        block_out_channels[i]
                        if i < len(block_out_channels) - 1 and j == k == 0
                        else 0
                    )
                    block = get_block(
                        block_type,
                        block_out_channels[i],
                        num_attention_heads[i],
                        c_skip=c_skip,
                        dropout=dropout[i],
                        self_attn=self_attn[i],
                        dtype=dtype,
                    )
                    up_block.append(block)
            self.up_blocks.append(up_block)

            if up_blocks_repeat_mappers is not None:
                block_repeat_mappers = nn.ModuleList()
                for _ in range(up_blocks_repeat_mappers[::-1][i] - 1):
                    block_repeat_mappers.append(
                        nn.Conv2d(
                            block_out_channels[i],
                            block_out_channels[i],
                            kernel_size=1,
                            dtype=dtype,
                        )
                    )
                self.up_repeat_mappers.append(block_repeat_mappers)

        # OUTPUT
        self.clf = nn.Sequential(
            SDCascadeLayerNorm(
                block_out_channels[0], elementwise_affine=False, eps=1e-6, dtype=dtype
            ),
            nn.Conv2d(
                block_out_channels[0],
                out_channels * (patch_size**2),
                kernel_size=1,
                dtype=dtype,
            ),
            nn.PixelShuffle(patch_size),
        )

        self.gradient_checkpointing = False

    def get_timestep_ratio_embedding(self, timestep_ratio: Tensor, max_positions=10000):
        r = ops.cast()(timestep_ratio * max_positions, "float32")
        half_dim = self.timestep_ratio_embedding_dim // 2

        emb = math.log(max_positions) / (half_dim - 1)
        emb = ops.arange(0, half_dim, 1)()
        emb = ops.cast()(emb, "float32")
        emb = ops.exp(emb * -emb)
        emb = ops.unsqueeze(1)(r) * ops.unsqueeze(0)(emb)
        emb = ops.concatenate()([ops.sin(emb), ops.cos(emb)], dim=1)

        if self.timestep_ratio_embedding_dim % 2 == 1:  # zero pad
            emb = ops.pad((0, 1), mode="constant")(emb)

        return ops.cast()(emb, dtype=timestep_ratio.dtype())

    def get_clip_embeddings(
        self,
        clip_txt_pooled: Tensor,
        clip_txt: Optional[Tensor] = None,
        clip_img: Optional[Tensor] = None,
    ):
        if len(ops.size()(clip_txt_pooled)) == 2:
            clip_txt_pool = ops.unsqueeze(1)(clip_txt_pooled)
        clip_txt_pool = ops.reshape()(
            self.clip_txt_pooled_mapper(clip_txt_pooled),
            [
                ops.size()(clip_txt_pooled, dim=0)._attrs["int_var"],
                ops.size()(clip_txt_pooled, dim=1)._attrs["int_var"] * self.clip_seq,
                -1,
            ],
        )
        if clip_txt is not None and clip_img is not None:
            clip_txt = self.clip_txt_mapper(clip_txt)
            if len(ops.size()(clip_img)) == 2:
                clip_img = ops.unsqueeze(1)(clip_img)
            clip_img = ops.reshape()(
                self.clip_img_mapper(clip_img),
                [
                    ops.size()(clip_img, dim=0)._attrs["int_var"],
                    ops.size()(clip_img, dim=1)._attrs["int_var"] * self.clip_seq,
                    -1,
                ],
            )
            clip = ops.concatenate()([clip_txt, clip_txt_pool, clip_img], dim=1)
        else:
            clip = clip_txt_pool
        return self.clip_norm(clip)

    def _down_encode(self, x, r_embed, clip):
        level_outputs = []
        block_group = zip(
            self.down_blocks, self.down_downscalers, self.down_repeat_mappers
        )

        for down_block, downscaler, repmap in block_group:
            x = downscaler(x)
            for i in range(len(repmap) + 1):
                for block in down_block:
                    if isinstance(block, SDCascadeResBlock):
                        x = block(x)
                    elif isinstance(block, SDCascadeAttnBlock):
                        x = block(x, clip)
                    elif isinstance(block, SDCascadeTimestepBlock):
                        x = block(x, r_embed)
                    else:
                        x = block(x)
                if i < len(repmap):
                    x = repmap[i](x)
            level_outputs.insert(0, x)
        return level_outputs

    def _up_decode(self, level_outputs, r_embed, clip):
        x = level_outputs[0]
        block_group = zip(self.up_blocks, self.up_upscalers, self.up_repeat_mappers)

        for i, (up_block, upscaler, repmap) in enumerate(block_group):
            for j in range(len(repmap) + 1):
                for k, block in enumerate(up_block):
                    if isinstance(block, SDCascadeResBlock):
                        skip = level_outputs[i] if k == 0 and i > 0 else None
                        if skip is not None and (
                            ops.size()(x, dim=1) != ops.size()(skip, dim=1)
                            or ops.size()(x, dim=2) != ops.size()(skip, dim=2)
                        ):
                            orig_type = x.dtype()
                            out = ops.size()(x)
                            out[1] = ops.size()(skip, dim=1)
                            out[2] = ops.size()(skip, dim=2)
                            out = [x._attrs["int_var"] for x in out]
                            out = Tensor(out)
                            x = ops.upsampling2d(
                                scale_factor=2.0, mode="bilinear", align_corners=True
                            )(ops.cast()(x, "float32"), out=out)
                            x = ops.cast()(x, orig_type)
                        x = block(x, skip)
                    elif isinstance(block, SDCascadeAttnBlock):
                        x = block(x, clip)
                    elif isinstance(block, SDCascadeTimestepBlock):
                        x = block(x, r_embed)
                    else:
                        x = block(x)
                if j < len(repmap):
                    x = repmap[j](x)
            x = upscaler(x)
        return x

    def forward(
        self,
        sample: Tensor,
        timestep_ratio: Tensor,
        clip_text_pooled: Tensor,
        clip_text: Optional[Tensor] = None,
        clip_img: Optional[Tensor] = None,
        effnet=None,
        pixels: Optional[Tensor] = None,
        sca=None,
        crp=None,
        return_dict=True,
    ):
        if pixels is None:
            pixels = ops.full()(
                [ops.size()(sample, dim=0)._attrs["int_var"], 3, 8, 8],
                fill_value=0.0,
                dtype=sample.dtype(),
            )

        # Process the conditioning embeddings
        timestep_ratio_embed = self.get_timestep_ratio_embedding(timestep_ratio)
        for c in self.timestep_conditioning_type:
            if c == "sca":
                cond = sca
            elif c == "crp":
                cond = crp
            else:
                cond = None
            t_cond = cond or ops.full()(
                [dim._attrs["int_var"] for dim in ops.size()(timestep_ratio)],
                fill_value=0.0,
                dtype=timestep_ratio_embed.dtype(),
            )
            timestep_ratio_embed = ops.concatenate()(
                [timestep_ratio_embed, self.get_timestep_ratio_embedding(t_cond)], dim=1
            )
        clip = self.get_clip_embeddings(
            clip_txt_pooled=clip_text_pooled, clip_txt=clip_text, clip_img=clip_img
        )

        # Model Blocks
        x = self.embedding(sample)
        if hasattr(self, "effnet_mapper") and effnet is not None:
            out = ops.size()(x)
            out[1] = ops.size()(x, dim=1)
            out[2] = ops.size()(x, dim=2)
            out = [x._attrs["int_var"] for x in out]
            out = Tensor(out)
            x = x + self.effnet_mapper(
                ops.upsampling2d(scale_factor=2.0, mode="bilinear", align_corners=True)(
                    effnet, out=out
                )
            )
        if hasattr(self, "pixels_mapper"):
            out = ops.size()(x)
            out[1] = ops.size()(x, dim=1)
            out[2] = ops.size()(x, dim=2)
            out = [x._attrs["int_var"] for x in out]
            out = Tensor(out)
            x = x + ops.upsampling2d(
                scale_factor=2.0, mode="bilinear", align_corners=True
            )(self.pixels_mapper(pixels), out=out)
        level_outputs = self._down_encode(x, timestep_ratio_embed, clip)
        x = self._up_decode(level_outputs, timestep_ratio_embed, clip)
        sample = self.clf(x)

        if not return_dict:
            return (sample,)
        return StableCascadeUNetOutput(sample=sample)
