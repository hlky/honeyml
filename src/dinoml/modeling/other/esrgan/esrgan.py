from typing import Annotated
from dinoml.compiler import ops
from dinoml.compiler.base import Tensor
from dinoml.frontend import nn
from dinoml.modeling.other.esrgan.rrdb import RRDB
from dinoml.utils.build_utils import Shape


class ESRGAN(nn.Module):
    """ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(
        self,
        num_in_ch: int,
        num_out_ch: int,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
        dtype: str = "float16",
    ):
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        if num_in_ch < 8:
            self.conv_first = nn.Conv2d(
                num_in_ch, num_feat, 3, 1, 1, dtype=dtype, few_channels=True
            )
        else:
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, dtype=dtype)
        self.body = nn.Sequential(
            *[
                RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch, dtype=dtype)
                for _ in range(num_block)
            ]
        )
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1, dtype=dtype)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, dtype=dtype)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, dtype=dtype)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1, dtype=dtype)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1, dtype=dtype)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def interpolate(self, tensor: Tensor):
        op = ops.upsampling2d(scale_factor=2, mode="nearest")
        out = op(tensor)
        return out

    def forward(
        self,
        x: Annotated[
            Tensor,
            (
                Shape(name="batch_size"),
                Shape(name="height"),
                Shape(name="width"),
                Shape(name="channels", config_name="num_in_ch"),
            ),
        ],
    ):
        if self.scale == 2:
            feat = ops.pixel_unshuffle(r=2)(x)
        elif self.scale == 1:
            feat = ops.pixel_unshuffle(r=4)(x)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(self.conv_up1(self.interpolate(feat)))
        feat = self.lrelu(self.conv_up2(self.interpolate(feat)))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out
