from honey.compiler.ops import (
    squeeze,
    transposed_conv2d,
    transposed_conv2d_bias,
    unsqueeze,
)
from honey.frontend import Tensor
from honey.frontend.nn.module import Module
from honey.frontend.nn.parameter import Parameter


class ConvTranspose1d(Module):
    r"""
    Applies a 1D transposed convolution operator over an input image composed of several input planes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        dtype: str = "float16",
        bias: bool = True,
    ):
        super().__init__()

        self.weight = Parameter(
            shape=[in_channels, kernel_size, out_channels // groups],
            dtype=dtype,
        )
        if bias:
            self.bias = Parameter(shape=[out_channels], dtype=dtype)
        else:
            self.bias = None
        fwd_func = transposed_conv2d_bias if bias else transposed_conv2d
        self.op = fwd_func(
            stride=(stride, 1), pad=(padding, 0), dilate=(dilation, 1), group=groups
        )

    def forward(self, x: Tensor) -> Tensor:
        xu = unsqueeze(dim=2)(x)
        wu = unsqueeze(dim=2)(self.weight.tensor())
        if self.bias is None:
            c2d = self.op(xu, wu)
        else:
            c2d = self.op(xu, wu, self.bias.tensor())
        return squeeze(dim=2)(c2d)
