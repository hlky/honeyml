from typing import Optional, Tuple

from dinoml.compiler import ops

from dinoml.frontend import nn, Tensor

from .normalization import RMSNorm
from .upsampling import upfirdn2d_native


class Downsample1D(nn.Module):
    """A 1D downsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        padding (`int`, default `1`):
            padding for the convolution.
        name (`str`, default `conv`):
            name of the downsampling 1D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
        name: str = "conv",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if use_conv:
            self.conv = nn.Conv1d(
                self.channels,
                self.out_channels,
                3,
                stride=stride,
                padding=padding,
                bias=True,
            )
        else:
            assert self.channels == self.out_channels
            self.conv = nn.AvgPool1d(kernel_size=stride, stride=stride)

    def forward(self, inputs: Tensor) -> Tensor:
        assert inputs.shape()[-1] == self.channels
        return self.conv(inputs)


class Downsample2D(nn.Module):
    """A 2D downsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        padding (`int`, default `1`):
            padding for the convolution.
        name (`str`, default `conv`):
            name of the downsampling 2D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
        name: str = "conv",
        kernel_size=3,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
        dtype: str = "float16",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name
        self.dtype = dtype

        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(
                channels, eps, elementwise_affine=elementwise_affine, dtype=dtype
            )
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(channels, eps, elementwise_affine, dtype)
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        if use_conv:
            conv = nn.Conv2d(
                self.channels,
                self.out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dtype=dtype,
                bias=bias,
            )
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool2d(kernel_size=stride, stride=stride, padding=0)

        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv = conv

    def forward(self, hidden_states: Tensor, *args, **kwargs) -> Tensor:
        assert hidden_states.shape()[-1] == self.channels

        if self.norm is not None:
            hidden_states = self.norm(hidden_states)

        if self.use_conv and self.padding == 0:
            # TODO
            padding = ops.full()([0, 1, 0, 0], 0.0, dtype=self.dtype)
            padding._attrs["shape"][0] = hidden_states._attrs["shape"][0]
            padding._attrs["shape"][2] = hidden_states._attrs["shape"][2]
            padding._attrs["shape"][3] = hidden_states._attrs["shape"][3]
            hidden_states = ops.concatenate()([hidden_states, padding], dim=1)
            padding = ops.full()([0, 0, 1, 0], 0.0, dtype=self.dtype)
            padding._attrs["shape"][0] = hidden_states._attrs["shape"][0]
            padding._attrs["shape"][1] = hidden_states._attrs["shape"][1]
            padding._attrs["shape"][3] = hidden_states._attrs["shape"][3]
            hidden_states = ops.concatenate()([hidden_states, padding], dim=2)

        assert hidden_states.shape()[-1] == self.channels

        hidden_states = self.conv(hidden_states)

        return hidden_states


class FirDownsample2D(nn.Module):
    """A 2D FIR downsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        fir_kernel (`tuple`, default `(1, 3, 3, 1)`):
            kernel for the FIR filter.
    """

    def __init__(
        self,
        channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        use_conv: bool = False,
        fir_kernel: Tuple[int, int, int, int] = (1, 3, 3, 1),
    ):
        super().__init__()
        out_channels = out_channels if out_channels else channels
        if use_conv:
            self.Conv2d_0 = nn.Conv2d(
                channels, out_channels, kernel_size=3, stride=1, padding=1
            )
        assert fir_kernel == (1, 3, 3, 1), "ops have fused `fir_kernel`"
        self.fir_kernel = fir_kernel
        self.use_conv = use_conv
        self.out_channels = out_channels

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self.use_conv:
            fir_padded = ops.fir_filter_pad2()(hidden_states)
            hidden_states = ops.conv2d(stride=2, bias=False)(
                fir_padded, self.Conv2d_0.weight.tensor()
            ) + ops.reshape()(self.Conv2d_0.bias.tensor(), [1, 1, 1, -1])
        else:
            hidden_states = ops.fir_downsample2d()(hidden_states)

        return hidden_states


# downsample/upsample layer used in k-upscaler, might be able to use FirDownsample2D/DirUpsample2D instead
class KDownsample2D(nn.Module):
    r"""A 2D K-downsampling layer.

    Parameters:
        pad_mode (`str`, *optional*, default to `"reflect"`): the padding mode to use.
    """

    def __init__(self, pad_mode: str = "reflect"):
        super().__init__()
        self.pad_mode = pad_mode
        self.pad = 1

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = ops.pad((self.pad,) * 4, self.pad_mode)(inputs)
        weight = ops.kdownsample2d_weight()(
            channels=inputs._attrs["shape"][-1], dtype=inputs.dtype()
        )
        return ops.conv2d(stride=2, bias=False)(inputs, weight)


class CogVideoXDownsample3D(nn.Module):
    # Todo: Wait for paper release.
    r"""
    A 3D Downsampling layer using in [CogVideoX]() by Tsinghua University & ZhipuAI

    Args:
        in_channels (`int`):
            Number of channels in the input image.
        out_channels (`int`):
            Number of channels produced by the convolution.
        kernel_size (`int`, defaults to `3`):
            Size of the convolving kernel.
        stride (`int`, defaults to `2`):
            Stride of the convolution.
        padding (`int`, defaults to `0`):
            Padding added to all four sides of the input.
        compress_time (`bool`, defaults to `False`):
            Whether or not to compress the time dimension.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 0,
        compress_time: bool = False,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.compress_time = compress_time

    def forward(self, x: Tensor) -> Tensor:
        if self.compress_time:
            batch_size, frames, height, width, channels = x._attrs["shape"]

            # (batch_size, frames, height, width, channels) -> (batch_size, height, width, channels, frames) -> (batch_size * height * width, channels, frames)
            x = ops.reshape()(ops.permute()(x, [0, 2, 3, 4, 1]), [-1, channels, frames])

            x = ops.avg_pool1d_compress_time()(x)

            # (batch_size * height * width, channels, frames // 2 or (frames - 1) // 2) -> (batch_size, height, width, channels, frames // 2 or (frames - 1) // 2) -> (batch_size, frames // 2 or (frames - 1) // 2, height, width, channels)
            x = ops.permute()(
                ops.reshape()(
                    x, [batch_size, height, width, channels, x._attrs["shape"][-1]]
                ),
                [0, 4, 1, 2, 3],
            )

        # Pad the tensor
        pad = (0, 1, 0, 1)
        x = ops.pad(pad, mode="constant", value=0.0)(x)
        batch_size, frames, height, width, channels = x._attrs["shape"]
        # (batch_size, frames, height, width, channels) -> (batch_size * frames, height, width, channels)
        x = ops.reshape()(x, [batch_size * frames, height, width, channels])
        x = self.conv(x)
        # (batch_size * frames, height, width, channels) -> (batch_size, frames, height, width, channels)
        x = ops.reshape()(
            x,
            [
                batch_size,
                frames,
                x._attrs["shape"][1],
                x._attrs["shape"][2],
                x._attrs["shape"][3],
            ],
        )
        return x


def downsample_2d(
    hidden_states: Tensor,
    kernel: Optional[Tensor] = None,
    factor: int = 2,
    gain: float = 1,
) -> Tensor:
    return ops.fir_downsample2d()(hidden_states)
