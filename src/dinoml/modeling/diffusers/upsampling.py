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
            conv = nn.ConvTranspose2dBias(
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

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
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

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # TODO(Suraj): Remove this cast once the issue is fixed in PyTorch
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = hidden_states.dtype()
        if dtype == "bfloat16":
            hidden_states = ops.cast()(hidden_states, "float32")

        # TODO: test this with DinoML kernel
        # # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        # if hidden_states.shape[0] >= 64:
        #     hidden_states = hidden_states.contiguous()

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

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == "bfloat16":
            hidden_states = ops.cast()(hidden_states, dtype)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
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

    def _upsample_2d(
        self,
        hidden_states: Tensor,
        weight: Optional[Tensor] = None,
        kernel: Optional[Tensor] = None,
        factor: int = 2,
        gain: float = 1,
    ) -> Tensor:
        """Fused `upsample_2d()` followed by `Conv2d()`.

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
                corresponds to nearest-neighbor upsampling.
            factor (`int`, *optional*): Integer upsampling factor (default: 2).
            gain (`float`, *optional*): Scaling factor for signal magnitude (default: 1.0).

        Returns:
            output (`Tensor`):
                Tensor of the shape `[N, H * factor, W * factor, C]`, and same
                datatype as `hidden_states`.
        """
        raise NotImplementedError("torch.outer, torch.flip")

        assert isinstance(factor, int) and factor >= 1

        # Setup filter kernel.
        if kernel is None:
            kernel = [1] * factor

        # setup kernel
        kernel = torch.tensor(kernel, dtype=torch.float32)
        if kernel.ndim == 1:
            kernel = torch.outer(kernel, kernel)
        kernel /= torch.sum(kernel)

        kernel = kernel * (gain * (factor**2))

        if self.use_conv:
            convH = weight.shape[2]
            convW = weight.shape[3]
            inC = weight.shape[1]

            pad_value = (kernel.shape[0] - factor) - (convW - 1)

            stride = (factor, factor)
            # Determine data dimensions.
            output_shape = (
                (hidden_states.shape[2] - 1) * factor + convH,
                (hidden_states.shape[3] - 1) * factor + convW,
            )
            output_padding = (
                output_shape[0] - (hidden_states.shape[2] - 1) * stride[0] - convH,
                output_shape[1] - (hidden_states.shape[3] - 1) * stride[1] - convW,
            )
            assert output_padding[0] >= 0 and output_padding[1] >= 0
            num_groups = hidden_states.shape[1] // inC

            # Transpose weights.
            weight = torch.reshape(weight, (num_groups, -1, inC, convH, convW))
            weight = torch.flip(weight, dims=[3, 4]).permute(0, 2, 1, 3, 4)
            weight = torch.reshape(weight, (num_groups * inC, -1, convH, convW))

            inverse_conv = F.conv_transpose2d(
                hidden_states,
                weight,
                stride=stride,
                output_padding=output_padding,
                padding=0,
            )

            output = upfirdn2d_native(
                inverse_conv,
                torch.tensor(kernel, device=inverse_conv.device),
                pad=((pad_value + 1) // 2 + factor - 1, pad_value // 2 + 1),
            )
        else:
            pad_value = kernel.shape[0] - factor
            output = upfirdn2d_native(
                hidden_states,
                torch.tensor(kernel, device=hidden_states.device),
                up=factor,
                pad=((pad_value + 1) // 2 + factor - 1, pad_value // 2),
            )

        return output

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self.use_conv:
            height = self._upsample_2d(
                hidden_states, self.Conv2d_0.weight, kernel=self.fir_kernel
            )
            height = height + self.Conv2d_0.bias.reshape(1, -1, 1, 1)
        else:
            height = self._upsample_2d(hidden_states, kernel=self.fir_kernel, factor=2)

        return height


class KUpsample2D(nn.Module):
    r"""A 2D K-upsampling layer.

    Parameters:
        pad_mode (`str`, *optional*, default to `"reflect"`): the padding mode to use.
    """

    def __init__(self, pad_mode: str = "reflect"):
        super().__init__()
        self.pad_mode = pad_mode
        """
        torch.tensor([[1 / 8, 3 / 8, 3 / 8, 1 / 8]]) * 2
        tensor([[0.2500, 0.7500, 0.7500, 0.2500]])
        kernel_1d.T @ kernel_1d
        """
        # NOTE/TODO: name may interact with others
        kernel_1d = Tensor([1, 4], name="kernel")
        self.pad = ops.size()(kernel_1d, dim=1)["int_var"].upper_bound() / 2 - 1
        self.kernel = kernel_1d

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError("weight assignment")
        inputs = ops.pad(((self.pad + 1) // 2,) * 4, mode=self.pad_mode)(inputs)
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
        return ops.transposed_conv2d(stride=2, pad=self.pad * 2 + 1, bias=False)(
            inputs, weight
        )


def upfirdn2d_native(
    tensor: Tensor,
    kernel: Tensor,
    up: int = 1,
    down: int = 1,
    pad: Tuple[int, int] = (0, 0),
) -> Tensor:
    raise NotImplementedError("torch.flip")
    up_x = up_y = up
    down_x = down_y = down
    pad_x0 = pad_y0 = pad[0]
    pad_x1 = pad_y1 = pad[1]

    _, channel, in_h, in_w = tensor.shape
    tensor = tensor.reshape(-1, in_h, in_w, 1)

    _, in_h, in_w, minor = tensor.shape
    kernel_h, kernel_w = kernel.shape

    out = tensor.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(
        out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )
    out = out.to(tensor.device)  # Move back to mps if necessary
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

    return out.view(-1, channel, out_h, out_w)


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
    raise NotImplementedError("`torch.outer`, `upfirdn2d_native`")
    assert isinstance(factor, int) and factor >= 1
    if kernel is None:
        kernel = [1] * factor

    kernel = torch.tensor(kernel, dtype=torch.float32)
    if kernel.ndim == 1:
        kernel = torch.outer(kernel, kernel)
    kernel /= torch.sum(kernel)

    kernel = kernel * (gain * (factor**2))
    pad_value = kernel.shape[0] - factor
    output = upfirdn2d_native(
        hidden_states,
        kernel.to(device=hidden_states.device),
        up=factor,
        pad=((pad_value + 1) // 2 + factor - 1, pad_value // 2),
    )
    return output
