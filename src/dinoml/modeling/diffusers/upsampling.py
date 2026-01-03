from typing import Optional, Tuple

from dinoml.compiler import ops

from dinoml.frontend import nn, Tensor

from .normalization import RMSNorm


class Upsample1D(nn.Module):
    """A 1D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        name (`str`, default `conv`):
            name of the upsampling 1D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: Optional[int] = None,
        name: str = "conv",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        self.conv = None
        if use_conv_transpose:
            self.conv = nn.ConvTranspose1d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            self.conv = nn.Conv1d(
                self.channels, self.out_channels, 3, padding=1, bias=True
            )

    def forward(self, inputs: Tensor) -> Tensor:
        assert inputs.shape()[-1] == self.channels
        if self.use_conv_transpose and self.conv is not None:
            return self.conv(inputs)

        outputs = ops.upsampling1d(scale_factor=2.0, mode="nearest")(inputs)

        if self.use_conv and self.conv is not None:
            outputs = self.conv(outputs)

        return outputs


class Upsample2D(nn.Module):
    """A 2D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        name (`str`, default `conv`):
            name of the upsampling 2D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: Optional[int] = None,
        name: str = "conv",
        kernel_size: Optional[int] = None,
        padding=1,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
        interpolate=True,
        dtype: str = "float16",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.interpolate = interpolate

        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(
                channels,
                eps,
                elementwise_affine=elementwise_affine,
                dtype=dtype,
            )
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(
                channels, eps, elementwise_affine=elementwise_affine, dtype=dtype
            )
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        conv = None
        if use_conv_transpose:
            if kernel_size is None:
                kernel_size = 4
            conv = nn.ConvTranspose2d(
                channels,
                self.out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                dtype=dtype,
            )
        elif use_conv:
            if kernel_size is None:
                kernel_size = 3
            conv = nn.Conv2d(
                self.channels,
                self.out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dtype=dtype,
            )

        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(
        self, hidden_states: Tensor, output_size: Optional[int] = None, *args, **kwargs
    ) -> Tensor:
        assert hidden_states.shape()[-1] == self.channels

        if self.norm is not None:
            hidden_states = self.norm(hidden_states)

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if self.interpolate:
            if output_size is None:
                hidden_states = ops.upsampling2d(scale_factor=2.0, mode="nearest")(
                    hidden_states
                )
            else:
                out = ops.size()(hidden_states)
                out[1] = output_size
                out[2] = output_size
                out = [x._attrs["int_var"] for x in out]
                out = Tensor(out)
                hidden_states = ops.upsampling2d(scale_factor=2.0, mode="nearest")(
                    hidden_states, out=out
                )

        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.Conv2d_0(hidden_states)

        return hidden_states


class FirUpsample2D(nn.Module):
    """A 2D FIR upsampling layer with an optional convolution.

    Parameters:
        channels (`int`, optional):
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
        self.use_conv = use_conv
        self.fir_kernel = fir_kernel
        self.out_channels = out_channels
        self.weight = None

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self.use_conv:
            if self.weight is None:
                self.weight = self.Conv2d_0.weight.tensor()
                _, H, W, I = self.weight.shape()
                num_groups = hidden_states.shape()[-1] / I
                self.weight = ops.reshape()(self.weight, (num_groups, -1, H, W, I))
                self.weight = ops.flip(dims=[2, 3])(self.weight)
                self.weight = ops.permute()(self.weight, [0, 4, 2, 3, 1])
                self.weight = ops.reshape()(self.weight, [num_groups * I, H, W, -1])
            height = ops.transposed_conv2d(stride=2, pad=0, bias=False)(
                hidden_states, self.weight
            )
            height = ops.fir_upsample2d()(height, up=1, pad0=1, pad1=1)
            height = height + ops.reshape()(self.Conv2d_0.bias.tensor(), [1, 1, 1, -1])
        else:
            height = ops.fir_upsample2d()(hidden_states)

        return height


class KUpsample2D(nn.Module):
    r"""A 2D K-upsampling layer.

    Parameters:
        pad_mode (`str`, *optional*, default to `"reflect"`): the padding mode to use.
    """

    def __init__(self, pad_mode: str = "reflect"):
        super().__init__()
        self.pad_mode = pad_mode
        self.pad = 1

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = ops.pad(((self.pad + 1) // 2,) * 4, mode=self.pad_mode)(inputs)
        weight = ops.kupsample2d_weight()(
            channels=inputs._attrs["shape"][-1], dtype=inputs.dtype()
        )
        return ops.transposed_conv2d(stride=2, pad=self.pad * 2 + 1, bias=False)(
            inputs, weight
        )


class CogVideoXUpsample3D(nn.Module):
    r"""
    A 3D Upsample layer using in CogVideoX by Tsinghua University & ZhipuAI # Todo: Wait for paper release.

    Args:
        in_channels (`int`):
            Number of channels in the input image.
        out_channels (`int`):
            Number of channels produced by the convolution.
        kernel_size (`int`, defaults to `3`):
            Size of the convolving kernel.
        stride (`int`, defaults to `1`):
            Stride of the convolution.
        padding (`int`, defaults to `1`):
            Padding added to all four sides of the input.
        compress_time (`bool`, defaults to `False`):
            Whether or not to compress the time dimension.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        compress_time: bool = False,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.compress_time = compress_time

    def forward(self, inputs: Tensor) -> Tensor:
        if self.compress_time:
            inputs = ops.upsampling3d_compress_time()(inputs)
        else:
            # only interpolate 2D
            b, t, h, w, c = inputs._attrs["shape"]
            inputs = ops.reshape()(inputs, [-1, h, w, c])
            inputs = ops.upsampling2d(scale_factor=2.0, mode="nearest")(inputs)
            _, h, w, _ = inputs._attrs["shape"]
            inputs = ops.reshape()(inputs, [b, t, h, w, c])

        b, t, h, w, c = inputs._attrs["shape"]
        inputs = ops.reshape()(inputs, [-1, h, w, c])
        inputs = self.conv(inputs)
        _, h, w, c = inputs._attrs["shape"]
        inputs = ops.reshape()(inputs, [b, t, h, w, c])

        return inputs


def upsample_2d(
    hidden_states: Tensor,
    kernel: Optional[Tensor] = None,
    factor: int = 2,
    gain: float = 1,
) -> Tensor:
    r"""Upsample2D a batch of 2D images with the given filter.
    Accepts a batch of 2D images of the shape `[N, H, W, C]` and upsamples each image with the given
    filter. The filter is normalized so that if the input pixels are constant, they will be scaled by the specified
    `gain`. Pixels outside the image are assumed to be zero, and the filter is padded with zeros so that its shape is
    a: multiple of the upsampling factor.

    Args:
        hidden_states (`Tensor`):
            Input tensor of the shape `[N, H, W, C]`.
        kernel (`Tensor`, *optional*):
            FIR filter of the shape `[firH, firW]` or `[firN]` (separable). The default is `[1] * factor`, which
            corresponds to nearest-neighbor upsampling.
        factor (`int`, *optional*, default to `2`):
            Integer upsampling factor.
        gain (`float`, *optional*, default to `1.0`):
            Scaling factor for signal magnitude (default: 1.0).

    Returns:
        output (`Tensor`):
            Tensor of the shape `[N, H * factor, W * factor, C]`
    """
    return ops.fir_upsample2d()(hidden_states)
