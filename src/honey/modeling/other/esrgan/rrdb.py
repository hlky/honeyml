from typing import List
from honey.compiler import ops
from honey.compiler.base import Tensor
from honey.frontend import nn


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2dBias(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2dBias(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2dBias(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2dBias(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2dBias(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def cat(self, tensors: List[Tensor], dim: int):
        out = ops.concatenate()(tensors, dim)
        return out

    def forward(self, x: Tensor):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(self.cat((x, x1), 3)))
        x3 = self.lrelu(self.conv3(self.cat((x, x1, x2), 3)))
        x4 = self.lrelu(self.conv4(self.cat((x, x1, x2, x3), 3)))
        x5 = self.conv5(self.cat((x, x1, x2, x3, x4), 3))
        # Empirically, we use 0.2 to scale the residual for better performance
        out = x5 * 0.2 + x
        return out


class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat: int, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: Tensor):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Empirically, we use 0.2 to scale the residual for better performance
        out = out * 0.2 + x
        return out
