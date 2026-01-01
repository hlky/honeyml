from typing import List

from dinoml.compiler import ops

from dinoml.frontend import nn, Tensor


class T2IAdapter(nn.Module):
    r"""
    A simple ResNet-like model that accepts images containing control signals such as keyposes and depth. The model
    generates multiple feature maps that are used as additional conditioning in [`UNet2DConditionModel`]. The model's
    architecture follows the original implementation of
    [Adapter](https://github.com/TencentARC/T2I-Adapter/blob/686de4681515662c0ac2ffa07bf5dda83af1038a/ldm/modules/encoders/adapter.py#L97)
     and
     [AdapterLight](https://github.com/TencentARC/T2I-Adapter/blob/686de4681515662c0ac2ffa07bf5dda83af1038a/ldm/modules/encoders/adapter.py#L235).

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the common methods, such as
    downloading or saving.

    Args:
        in_channels (`int`, *optional*, defaults to `3`):
            The number of channels in the adapter's input (*control image*). Set it to 1 if you're using a gray scale
            image.
        channels (`List[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The number of channels in each downsample block's output hidden state. The `len(block_out_channels)`
            determines the number of downsample blocks in the adapter.
        num_res_blocks (`int`, *optional*, defaults to `2`):
            Number of ResNet blocks in each downsample block.
        downscale_factor (`int`, *optional*, defaults to `8`):
            A factor that determines the total downscale factor of the Adapter.
        adapter_type (`str`, *optional*, defaults to `full_adapter`):
            Adapter type (`full_adapter` or `full_adapter_xl` or `light_adapter`) to use.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [320, 640, 1280, 1280],
        num_res_blocks: int = 2,
        downscale_factor: int = 8,
        adapter_type: str = "full_adapter",
        dtype: str = "float16",
    ):
        super().__init__()

        if adapter_type == "full_adapter":
            self.adapter = FullAdapter(
                in_channels, channels, num_res_blocks, downscale_factor, dtype=dtype
            )
        elif adapter_type == "full_adapter_xl":
            self.adapter = FullAdapterXL(
                in_channels, channels, num_res_blocks, downscale_factor, dtype=dtype
            )
        elif adapter_type == "light_adapter":
            self.adapter = LightAdapter(
                in_channels, channels, num_res_blocks, downscale_factor, dtype=dtype
            )
        else:
            raise ValueError(
                f"Unsupported adapter_type: '{adapter_type}'. Choose either 'full_adapter' or "
                "'full_adapter_xl' or 'light_adapter'."
            )

    def forward(self, x: Tensor) -> List[Tensor]:
        r"""
        This function processes the input tensor `x` through the adapter model and returns a list of feature tensors,
        each representing information extracted at a different scale from the input. The length of the list is
        determined by the number of downsample blocks in the Adapter, as specified by the `channels` and
        `num_res_blocks` parameters during initialization.
        """
        return self.adapter(x)

    @property
    def total_downscale_factor(self):
        return self.adapter.total_downscale_factor

    @property
    def downscale_factor(self):
        """The downscale factor applied in the T2I-Adapter's initial pixel unshuffle operation. If an input image's dimensions are
        not evenly divisible by the downscale_factor then an exception will be raised.
        """
        return self.adapter.unshuffle.downscale_factor


# full adapter


class FullAdapter(nn.Module):
    r"""
    See [`T2IAdapter`] for more information.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [320, 640, 1280, 1280],
        num_res_blocks: int = 2,
        downscale_factor: int = 8,
        dtype: str = "float16",
    ):
        super().__init__()

        in_channels = in_channels * downscale_factor**2

        self.unshuffle = nn.PixelUnshuffle(downscale_factor)
        self.conv_in = nn.Conv2d(
            in_channels, channels[0], kernel_size=3, padding=1, dtype=dtype
        )

        self.body = nn.ModuleList(
            [
                AdapterBlock(channels[0], channels[0], num_res_blocks, dtype=dtype),
                *[
                    AdapterBlock(
                        channels[i - 1],
                        channels[i],
                        num_res_blocks,
                        down=True,
                        dtype=dtype,
                    )
                    for i in range(1, len(channels))
                ],
            ]
        )

        self.total_downscale_factor = downscale_factor * 2 ** (len(channels) - 1)

    def forward(self, x: Tensor) -> List[Tensor]:
        r"""
        This method processes the input tensor `x` through the FullAdapter model and performs operations including
        pixel unshuffling, convolution, and a stack of AdapterBlocks. It returns a list of feature tensors, each
        capturing information at a different stage of processing within the FullAdapter model. The number of feature
        tensors in the list is determined by the number of downsample blocks specified during initialization.
        """
        x = self.unshuffle(x)
        x = self.conv_in(x)

        features = []

        for block in self.body:
            x = block(x)
            features.append(x)

        return features


class FullAdapterXL(nn.Module):
    r"""
    See [`T2IAdapter`] for more information.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [320, 640, 1280, 1280],
        num_res_blocks: int = 2,
        downscale_factor: int = 16,
        dtype: str = "float16",
    ):
        super().__init__()

        in_channels = in_channels * downscale_factor**2

        self.unshuffle = nn.PixelUnshuffle(downscale_factor)
        self.conv_in = nn.Conv2d(
            in_channels, channels[0], kernel_size=3, padding=1, dtype=dtype
        )

        self.body = []
        # blocks to extract XL features with dimensions of [320, 64, 64], [640, 64, 64], [1280, 32, 32], [1280, 32, 32]
        for i in range(len(channels)):
            if i == 1:
                self.body.append(
                    AdapterBlock(
                        channels[i - 1], channels[i], num_res_blocks, dtype=dtype
                    )
                )
            elif i == 2:
                self.body.append(
                    AdapterBlock(
                        channels[i - 1],
                        channels[i],
                        num_res_blocks,
                        down=True,
                        dtype=dtype,
                    )
                )
            else:
                self.body.append(
                    AdapterBlock(channels[i], channels[i], num_res_blocks, dtype=dtype)
                )

        self.body = nn.ModuleList(self.body)
        # XL has only one downsampling AdapterBlock.
        self.total_downscale_factor = downscale_factor * 2

    def forward(self, x: Tensor) -> List[Tensor]:
        r"""
        This method takes the tensor x as input and processes it through FullAdapterXL model. It consists of operations
        including unshuffling pixels, applying convolution layer and appending each block into list of feature tensors.
        """
        x = self.unshuffle(x)
        x = self.conv_in(x)

        features = []

        for block in self.body:
            x = block(x)
            features.append(x)

        return features


class AdapterBlock(nn.Module):
    r"""
    An AdapterBlock is a helper model that contains multiple ResNet-like blocks. It is used in the `FullAdapter` and
    `FullAdapterXL` models.

    Args:
        in_channels (`int`):
            Number of channels of AdapterBlock's input.
        out_channels (`int`):
            Number of channels of AdapterBlock's output.
        num_res_blocks (`int`):
            Number of ResNet blocks in the AdapterBlock.
        down (`bool`, *optional*, defaults to `False`):
            If `True`, perform downsampling on AdapterBlock's input.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int,
        down: bool = False,
        dtype: str = "float16",
    ):
        super().__init__()

        self.downsample = None
        if down:
            # ops TODO: AvgPool2d ceil_mode
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.in_conv = None
        if in_channels != out_channels:
            self.in_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, dtype=dtype
            )

        self.resnets = nn.Sequential(
            *[
                AdapterResnetBlock(out_channels, dtype=dtype)
                for _ in range(num_res_blocks)
            ],
        )

    def forward(self, x: Tensor) -> Tensor:
        r"""
        This method takes tensor x as input and performs operations downsampling and convolutional layers if the
        self.downsample and self.in_conv properties of AdapterBlock model are specified. Then it applies a series of
        residual blocks to the input tensor.
        """
        if self.downsample is not None:
            x = self.downsample(x)

        if self.in_conv is not None:
            x = self.in_conv(x)

        x = self.resnets(x)

        return x


class AdapterResnetBlock(nn.Module):
    r"""
    An `AdapterResnetBlock` is a helper model that implements a ResNet-like block.

    Args:
        channels (`int`):
            Number of channels of AdapterResnetBlock's input and output.
    """

    def __init__(self, channels: int, dtype: str = "float16"):
        super().__init__()
        self.block1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, dtype=dtype
        )
        self.act = ops.relu
        self.block2 = nn.Conv2d(channels, channels, kernel_size=1, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        r"""
        This method takes input tensor x and applies a convolutional layer, ReLU activation, and another convolutional
        layer on the input tensor. It returns addition with the input tensor.
        """

        h = self.act(self.block1(x))
        h = self.block2(h)

        return h + x


# light adapter


class LightAdapter(nn.Module):
    r"""
    See [`T2IAdapter`] for more information.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [320, 640, 1280],
        num_res_blocks: int = 4,
        downscale_factor: int = 8,
        dtype: str = "float16",
    ):
        super().__init__()

        in_channels = in_channels * downscale_factor**2

        self.unshuffle = nn.PixelUnshuffle(downscale_factor)

        self.body = nn.ModuleList(
            [
                LightAdapterBlock(
                    in_channels, channels[0], num_res_blocks, dtype=dtype
                ),
                *[
                    LightAdapterBlock(
                        channels[i],
                        channels[i + 1],
                        num_res_blocks,
                        down=True,
                        dtype=dtype,
                    )
                    for i in range(len(channels) - 1)
                ],
                LightAdapterBlock(
                    channels[-1], channels[-1], num_res_blocks, down=True, dtype=dtype
                ),
            ]
        )

        self.total_downscale_factor = downscale_factor * (2 ** len(channels))

    def forward(self, x: Tensor) -> List[Tensor]:
        r"""
        This method takes the input tensor x and performs downscaling and appends it in list of feature tensors. Each
        feature tensor corresponds to a different level of processing within the LightAdapter.
        """
        x = self.unshuffle(x)

        features = []

        for block in self.body:
            x = block(x)
            features.append(x)

        return features


class LightAdapterBlock(nn.Module):
    r"""
    A `LightAdapterBlock` is a helper model that contains multiple `LightAdapterResnetBlocks`. It is used in the
    `LightAdapter` model.

    Args:
        in_channels (`int`):
            Number of channels of LightAdapterBlock's input.
        out_channels (`int`):
            Number of channels of LightAdapterBlock's output.
        num_res_blocks (`int`):
            Number of LightAdapterResnetBlocks in the LightAdapterBlock.
        down (`bool`, *optional*, defaults to `False`):
            If `True`, perform downsampling on LightAdapterBlock's input.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int,
        down: bool = False,
        dtype: str = "float16",
    ):
        super().__init__()
        mid_channels = out_channels // 4

        self.downsample = None
        if down:
            # ops TODO: AvgPool2d ceil_mode
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.in_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, dtype=dtype)
        self.resnets = nn.Sequential(
            *[
                LightAdapterResnetBlock(mid_channels, dtype=dtype)
                for _ in range(num_res_blocks)
            ]
        )
        self.out_conv = nn.Conv2d(
            mid_channels, out_channels, kernel_size=1, dtype=dtype
        )

    def forward(self, x: Tensor) -> Tensor:
        r"""
        This method takes tensor x as input and performs downsampling if required. Then it applies in convolution
        layer, a sequence of residual blocks, and out convolutional layer.
        """
        if self.downsample is not None:
            x = self.downsample(x)

        x = self.in_conv(x)
        x = self.resnets(x)
        x = self.out_conv(x)

        return x


class LightAdapterResnetBlock(nn.Module):
    """
    A `LightAdapterResnetBlock` is a helper model that implements a ResNet-like block with a slightly different
    architecture than `AdapterResnetBlock`.

    Args:
        channels (`int`):
            Number of channels of LightAdapterResnetBlock's input and output.
    """

    def __init__(self, channels: int, dtype: str = "float16"):
        super().__init__()
        self.block1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, dtype=dtype
        )
        self.act = ops.relu
        self.block2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, dtype=dtype
        )

    def forward(self, x: Tensor) -> Tensor:
        r"""
        This function takes input tensor x and processes it through one convolutional layer, ReLU activation, and
        another convolutional layer and adds it to input tensor.
        """

        h = self.act(self.block1(x))
        h = self.block2(h)

        return h + x
