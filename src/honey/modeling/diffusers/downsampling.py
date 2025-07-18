from typing import Optional, Tuple

from honey.compiler import ops

from honey.frontend import nn, Tensor

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
            )
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool2d(kernel_size=stride, stride=stride, padding=0)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
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
        self.fir_kernel = fir_kernel
        self.use_conv = use_conv
        self.out_channels = out_channels

    def _downsample_2d(
        self,
        hidden_states: Tensor,
        weight: Optional[Tensor] = None,
        kernel: Optional[Tensor] = None,
        factor: int = 2,
        gain: float = 1,
    ) -> Tensor:
        """Fused `Conv2d()` followed by `downsample_2d()`.
        Padding is performed only once at the beginning, not between the operations. The fused op is considerably more
        efficient.

        Args:
            hidden_states (`Tensor`):
                Input tensor of the shape `[N, H, W, C]`.
            weight (`Tensor`, *optional*):
                Weight tensor of the shape `[filterH, filterW, inChannels, outChannels]`. Grouped convolution can be
                performed by `inChannels = x.shape[0] // numGroups`.
            kernel (`Tensor`, *optional*):
                FIR filter of the shape `[firH, firW]` or `[firN]` (separable). The default is `[1] * factor`, which
                corresponds to average pooling.
            factor (`int`, *optional*, default to `2`):
                Integer downsampling factor.
            gain (`float`, *optional*, default to `1.0`):
                Scaling factor for signal magnitude.

        Returns:
            output (`Tensor`):
                Tensor of the shape `[N, H // factor, W // factor, C]`, and same
                datatype as `x`.
        """
        raise NotImplementedError("`torch.outer`, `upfirdn2d_native`")
        assert isinstance(factor, int) and factor >= 1
        if kernel is None:
            kernel = [1] * factor

        # setup kernel
        kernel = torch.tensor(kernel, dtype=torch.float32)
        if kernel.ndim == 1:
            kernel = torch.outer(kernel, kernel)
        kernel /= ops.reduce_sum()(kernel)

        kernel = kernel * gain

        if self.use_conv:
            _, _, convH, convW = weight.shape
            pad_value = (kernel.shape[0] - factor) + (convW - 1)
            stride_value = [factor, factor]
            upfirdn_input = upfirdn2d_native(
                hidden_states,
                torch.tensor(kernel, device=hidden_states.device),
                pad=((pad_value + 1) // 2, pad_value // 2),
            )
            output = ops.conv2d(stride=stride_value, pad=0, bias=False)(upfirdn_input, weight)
        else:
            pad_value = kernel.shape[0] - factor
            output = upfirdn2d_native(
                hidden_states,
                torch.tensor(kernel, device=hidden_states.device),
                down=factor,
                pad=((pad_value + 1) // 2, pad_value // 2),
            )

        return output

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self.use_conv:
            downsample_input = self._downsample_2d(
                hidden_states,
                weight=self.Conv2d_0.weight.tensor(),
                kernel=self.fir_kernel,
            )
            hidden_states = downsample_input + ops.reshape()(
                self.Conv2d_0.bias.tensor(), [1, -1, 1, 1]
            )
        else:
            hidden_states = self._downsample_2d(
                hidden_states, kernel=self.fir_kernel, factor=2
            )

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
        # TODO: kernel
        """
        torch.tensor([[1 / 8, 3 / 8, 3 / 8, 1 / 8]])
        tensor([[0.1250, 0.3750, 0.3750, 0.1250]])
        kernel_1d.T @ kernel_1d
        """
        kernel_1d = Tensor([1, 4], name="kernel")
        self.pad = ops.size()(kernel_1d, dim=1)["int_var"].upper_bound() / 2 - 1
        self.kernel = kernel_1d

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError(f"weight assignment")
        inputs = ops.pad((self.pad,) * 4, mode=self.pad_mode)(inputs)
        inputs_dim1 = ops.size()(inputs, dim=1)
        kernel_dim0, kernel_dim1 = ops.size()(self.kernel)
        weight = ops.full()(
            inputs_dim1,
            inputs_dim1,
            kernel_dim0,
            kernel_dim1,
            fill_value=0.0,
            dtype=inputs.dtype(),
        )
        indices = ops.arange(0, inputs_dim1["int_var"], 1)()
        kernel = ops.expand()(
            ops.unsqueeze(0)(ops.cast()(self.kernel, weight.dtype())),
            inputs_dim1,
            -1,
            -1,
        )
        # TODO
        weight[indices, indices] = kernel
        return ops.conv2d(stride=2, pad=0, bias=False)(inputs, weight)


def downsample_2d(
    hidden_states: Tensor,
    kernel: Optional[Tensor] = None,
    factor: int = 2,
    gain: float = 1,
) -> Tensor:
    r"""Downsample2D a batch of 2D images with the given filter.
    Accepts a batch of 2D images of the shape `[N, H, W, C]` and downsamples each image with the
    given filter. The filter is normalized so that if the input pixels are constant, they will be scaled by the
    specified `gain`. Pixels outside the image are assumed to be zero, and the filter is padded with zeros so that its
    shape is a multiple of the downsampling factor.

    Args:
        hidden_states (`Tensor`)
            Input tensor of the shape `[N, H, W, C]`.
        kernel (`Tensor`, *optional*):
            FIR filter of the shape `[firH, firW]` or `[firN]` (separable). The default is `[1] * factor`, which
            corresponds to average pooling.
        factor (`int`, *optional*, default to `2`):
            Integer downsampling factor.
        gain (`float`, *optional*, default to `1.0`):
            Scaling factor for signal magnitude.

    Returns:
        output (`Tensor`):
            Tensor of the shape `[N, H // factor, W // factor, C]`
    """
    raise NotImplementedError("`torch.outer`, `upfirdn2d_native`")
    assert isinstance(factor, int) and factor >= 1
    if kernel is None:
        kernel = [1] * factor

    kernel = torch.tensor(kernel, dtype=torch.float32)
    if kernel.ndim == 1:
        kernel = torch.outer(kernel, kernel)
    kernel /= torch.sum(kernel)

    kernel = kernel * gain
    pad_value = kernel.shape[0] - factor
    output = upfirdn2d_native(
        hidden_states,
        kernel.to(device=hidden_states.device),
        down=factor,
        pad=((pad_value + 1) // 2, pad_value // 2),
    )
    return output
