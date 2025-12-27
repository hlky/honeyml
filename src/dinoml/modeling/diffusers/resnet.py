import math
from functools import partial
from typing import Annotated, cast, Optional, Tuple, Union

from dinoml.compiler import ops

from dinoml.compiler.base import IntVarTensor

from dinoml.frontend import IntVar, nn, Tensor

from .activations import get_activation
from .attention_processor import SpatialNorm
from .downsampling import (  # noqa
    Downsample1D,
    Downsample2D,
    downsample_2d,
    FirDownsample2D,
    KDownsample2D,
)
from .embeddings import SiLU
from .normalization import AdaGroupNorm
from .upsampling import (  # noqa
    FirUpsample2D,
    KUpsample2D,
    upfirdn2d_native,
    Upsample1D,
    Upsample2D,
    upsample_2d,
)
from ...utils.build_utils import Shape, DimAdd, DimDiv, DimMul, DimSub


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def get_shape(x):
    shape = [
        (
            it.value()
            if not isinstance(it, IntVar)
            else [it.lower_bound(), it.upper_bound()]
        )
        for it in x._attrs["shape"]
    ]
    return shape


class ResnetBlockCondNorm2D(nn.Module):
    r"""
    A Resnet block that use normalization layer that incorporate conditioning information.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
        temb_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
        groups (`int`, *optional*, default to `32`): The number of groups to use for the first normalization layer.
        groups_out (`int`, *optional*, default to None):
            The number of groups to use for the second normalization layer. if set to None, same as `groups`.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
        non_linearity (`str`, *optional*, default to `"swish"`): the activation function to use.
        time_embedding_norm (`str`, *optional*, default to `"ada_group"` ):
            The normalization layer for time embedding `temb`. Currently only support "ada_group" or "spatial".
        kernel (`Tensor`, optional, default to None): FIR filter, see
            [`~models.resnet.FirUpsample2D`] and [`~models.resnet.FirDownsample2D`].
        output_scale_factor (`float`, *optional*, default to be `1.0`): the scale factor to use for the output.
        use_in_shortcut (`bool`, *optional*, default to `True`):
            If `True`, add a 1x1 nn.conv2d layer for skip-connection.
        up (`bool`, *optional*, default to `False`): If `True`, add an upsample layer.
        down (`bool`, *optional*, default to `False`): If `True`, add a downsample layer.
        conv_shortcut_bias (`bool`, *optional*, default to `True`):  If `True`, adds a learnable bias to the
            `conv_shortcut` output.
        conv_2d_out_channels (`int`, *optional*, default to `None`): the number of channels in the output.
            If None, same as `out_channels`.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        groups_out: Optional[int] = None,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        time_embedding_norm: str = "ada_group",  # ada_group, spatial
        output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None,
        up: bool = False,
        down: bool = False,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: Optional[int] = None,
        dtype: str = "float16",
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm

        if groups_out is None:
            groups_out = groups

        if self.time_embedding_norm == "ada_group":  # ada_group
            self.norm1 = AdaGroupNorm(
                temb_channels, in_channels, groups, eps=eps, dtype=dtype
            )
        elif self.time_embedding_norm == "spatial":
            self.norm1 = SpatialNorm(in_channels, temb_channels, dtype=dtype)
        else:
            raise ValueError(
                f" unsupported time_embedding_norm: {self.time_embedding_norm}"
            )

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, dtype=dtype
        )

        if self.time_embedding_norm == "ada_group":  # ada_group
            self.norm2 = AdaGroupNorm(
                temb_channels, out_channels, groups_out, eps=eps, dtype=dtype
            )
        elif self.time_embedding_norm == "spatial":  # spatial
            self.norm2 = SpatialNorm(out_channels, temb_channels, dtype=dtype)
        else:
            raise ValueError(
                f" unsupported time_embedding_norm: {self.time_embedding_norm}"
            )

        self.dropout = nn.Dropout(dropout)

        conv_2d_out_channels = conv_2d_out_channels or out_channels
        self.conv2 = nn.Conv2d(
            out_channels,
            conv_2d_out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dtype=dtype,
        )

        self.nonlinearity = get_activation(non_linearity)

        self.upsample = self.downsample = None
        if self.up:
            self.upsample = Upsample2D(in_channels, use_conv=False, dtype=dtype)
        elif self.down:
            self.downsample = Downsample2D(
                in_channels, use_conv=False, padding=1, name="op", dtype=dtype
            )

        self.use_in_shortcut = (
            self.in_channels != conv_2d_out_channels
            if use_in_shortcut is None
            else use_in_shortcut
        )

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels,
                conv_2d_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dtype=dtype,
                bias=conv_shortcut_bias,
            )

    def forward(self, input_tensor: Tensor, temb: Tensor, *args, **kwargs) -> Tensor:
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states, temb)

        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)

        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states, temb)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


class ResnetBlock2D(nn.Module):
    r"""
    A Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
        temb_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
        groups (`int`, *optional*, default to `32`): The number of groups to use for the first normalization layer.
        groups_out (`int`, *optional*, default to None):
            The number of groups to use for the second normalization layer. if set to None, same as `groups`.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
        non_linearity (`str`, *optional*, default to `"swish"`): the activation function to use.
        time_embedding_norm (`str`, *optional*, default to `"default"` ): Time scale shift config.
            By default, apply timestep embedding conditioning with a simple shift mechanism. Choose "scale_shift" for a
            stronger conditioning with scale and shift.
        kernel (`Tensor`, optional, default to None): FIR filter, see
            [`~models.resnet.FirUpsample2D`] and [`~models.resnet.FirDownsample2D`].
        output_scale_factor (`float`, *optional*, default to be `1.0`): the scale factor to use for the output.
        use_in_shortcut (`bool`, *optional*, default to `True`):
            If `True`, add a 1x1 nn.conv2d layer for skip-connection.
        up (`bool`, *optional*, default to `False`): If `True`, add an upsample layer.
        down (`bool`, *optional*, default to `False`): If `True`, add a downsample layer.
        conv_shortcut_bias (`bool`, *optional*, default to `True`):  If `True`, adds a learnable bias to the
            `conv_shortcut` output.
        conv_2d_out_channels (`int`, *optional*, default to `None`): the number of channels in the output.
            If None, same as `out_channels`.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        groups_out: Optional[int] = None,
        pre_norm: bool = True,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        skip_time_act: bool = False,
        time_embedding_norm: str = "default",  # default, scale_shift,
        kernel: Optional[Tensor] = None,
        output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None,
        up: bool = False,
        down: bool = False,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: Optional[int] = None,
        dtype: str = "float16",
    ):
        super().__init__()
        if time_embedding_norm == "ada_group":
            raise ValueError(
                "This class cannot be used with `time_embedding_norm==ada_group`, please use `ResnetBlockCondNorm2D` instead",
            )
        if time_embedding_norm == "spatial":
            raise ValueError(
                "This class cannot be used with `time_embedding_norm==spatial`, please use `ResnetBlockCondNorm2D` instead",
            )

        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm
        self.skip_time_act = skip_time_act
        self._non_linearity = non_linearity

        if groups_out is None:
            groups_out = groups

        self.norm1 = nn.GroupNorm(
            num_groups=groups,
            num_channels=in_channels,
            eps=eps,
            affine=True,
            use_swish=True
            if (non_linearity == "swish" or non_linearity == "silu")
            else False,
            dtype=dtype,
        )

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, dtype=dtype
        )

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                self.time_emb_proj = nn.Linear(temb_channels, out_channels, dtype=dtype)
            elif self.time_embedding_norm == "scale_shift":
                self.time_emb_proj = nn.Linear(
                    temb_channels, 2 * out_channels, dtype=dtype
                )
            else:
                raise ValueError(
                    f"unknown time_embedding_norm : {self.time_embedding_norm} "
                )
        else:
            self.time_emb_proj = None

        self.norm2 = nn.GroupNorm(
            num_groups=groups_out,
            num_channels=out_channels,
            eps=eps,
            affine=True,
            use_swish=True
            if (non_linearity == "swish" or non_linearity == "silu")
            else False,
            dtype=dtype,
        )

        self.dropout = nn.Dropout(dropout)
        conv_2d_out_channels = conv_2d_out_channels or out_channels
        self.conv2 = nn.Conv2d(
            out_channels,
            conv_2d_out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dtype=dtype,
        )

        self.nonlinearity = get_activation(non_linearity)

        self.upsample = self.downsample = None
        if self.up:
            if kernel == "fir":
                raise NotImplementedError("'fir' upsample_2d not implemented.")
                fir_kernel = (1, 3, 3, 1)
                self.upsample = lambda x: upsample_2d(x, kernel=fir_kernel)
            elif kernel == "sde_vp":
                self.upsample = partial(
                    nn.upsampling2d(scale_factor=2.0, mode="nearest")
                )
            else:
                self.upsample = Upsample2D(in_channels, use_conv=False, dtype=dtype)
        elif self.down:
            if kernel == "fir":
                raise NotImplementedError("'fir' downsample_2d not implemented.")
                fir_kernel = (1, 3, 3, 1)
                self.downsample = lambda x: downsample_2d(x, kernel=fir_kernel)
            elif kernel == "sde_vp":
                self.downsample = partial(nn.avg_pool2d(kernel_size=2, stride=2, pad=0))
            else:
                self.downsample = Downsample2D(
                    in_channels, use_conv=False, padding=1, name="op", dtype=dtype
                )

        self.use_in_shortcut = (
            self.in_channels != conv_2d_out_channels
            if use_in_shortcut is None
            else use_in_shortcut
        )

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels,
                conv_2d_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dtype=dtype,
                bias=conv_shortcut_bias,
            )

    def forward(
        self,
        input_tensor: Annotated[
            Tensor,
            (
                Shape(name="batch_size"),
                Shape(name="height"),
                Shape(name="width"),
                Shape(name="channels", config_name="in_channels"),
            ),
        ],
        temb: Annotated[
            Tensor,
            (
                Shape(name="batch_size"),
                Shape(name="temb_channels", config_name="temb_channels"),
            ),
        ] = None,
        *args,
        **kwargs,
    ) -> Tensor:
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        if self._non_linearity != "swish" and self._non_linearity != "silu":
            hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = ops.unsqueeze(1)(ops.unsqueeze(1)(self.time_emb_proj(temb)))

        if self.time_embedding_norm == "default":
            if temb is not None:
                hidden_states = hidden_states + temb
            hidden_states = self.norm2(hidden_states)
        elif self.time_embedding_norm == "scale_shift":
            if temb is None:
                raise ValueError(
                    f" `temb` should not be None when `time_embedding_norm` is {self.time_embedding_norm}"
                )
            time_scale, time_shift = ops.chunk()(temb, 2, dim=-1)
            hidden_states = self.norm2(hidden_states)
            hidden_states = hidden_states * (1 + time_scale) + time_shift
        else:
            hidden_states = self.norm2(hidden_states)

        if self._non_linearity != "swish" and self._non_linearity != "silu":
            hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = input_tensor + hidden_states
        if self.output_scale_factor != 1.0:
            output_tensor = output_tensor / self.output_scale_factor

        return output_tensor


def rearrange_dims(tensor: Tensor) -> Tensor:
    shape = ops.size()(tensor)
    if len(shape) == 2:
        return ops.unsqueeze(1)(tensor)
    if len(shape) == 3:
        return ops.unsqueeze(2)(tensor)
    elif len(shape) == 4:
        return ops.dynamic_slice()(
            tensor, start_indices=[0, 0, 0, 0], end_indices=[None, None, 1, None]
        )
    else:
        raise ValueError(f"`len(tensor)`: {len(tensor)} has to be 2, 3 or 4.")


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish

    Parameters:
        inp_channels (`int`): Number of input channels.
        out_channels (`int`): Number of output channels.
        kernel_size (`int` or `tuple`): Size of the convolving kernel.
        n_groups (`int`, default `8`): Number of groups to separate the channels into.
        activation (`str`, defaults to `mish`): Name of the activation function.
    """

    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        n_groups: int = 8,
        activation: str = "mish",
        dtype: str = "float16",
    ):
        super().__init__()

        self.conv1d = nn.Conv1d(
            inp_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
            dtype=dtype,
        )
        self.group_norm = nn.GroupNorm(n_groups, out_channels, dtype=dtype)
        self.mish = get_activation(activation)

    def forward(self, inputs: Tensor) -> Tensor:
        intermediate_repr = self.conv1d(inputs)
        intermediate_repr = rearrange_dims(intermediate_repr)
        intermediate_repr = self.group_norm(intermediate_repr)
        intermediate_repr = rearrange_dims(intermediate_repr)
        output = self.mish(intermediate_repr)
        return ops.squeeze(2)(output)


class ResidualTemporalBlock1D(nn.Module):
    """
    Residual 1D block with temporal convolutions.

    Parameters:
        inp_channels (`int`): Number of input channels.
        out_channels (`int`): Number of output channels.
        embed_dim (`int`): Embedding dimension.
        kernel_size (`int` or `tuple`): Size of the convolving kernel.
        activation (`str`, defaults `mish`): It is possible to choose the right activation function.
    """

    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        embed_dim: int,
        kernel_size: Union[int, Tuple[int, int]] = 5,
        activation: str = "mish",
        dtype: str = "float16",
    ):
        super().__init__()
        self.conv_in = Conv1dBlock(inp_channels, out_channels, kernel_size, dtype=dtype)
        self.conv_out = Conv1dBlock(
            out_channels, out_channels, kernel_size, dtype=dtype
        )

        self.time_emb_act = get_activation(activation)
        self.time_emb = nn.Linear(embed_dim, out_channels, dtype=dtype)

        self.residual_conv = (
            nn.Conv1d(inp_channels, out_channels, 1, dtype=dtype)
            if inp_channels != out_channels
            else nn.Identity(dtype=dtype)
        )

    def forward(self, inputs: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            inputs : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]

        returns:
            out : [ batch_size x out_channels x horizon ]
        """
        t = self.time_emb_act(t)
        t = self.time_emb(t)
        out = self.conv_in(inputs) + rearrange_dims(t)
        out = self.conv_out(out)
        return out + self.residual_conv(inputs)


class TemporalConvLayer(nn.Module):
    """
    Temporal convolutional layer that can be used for video (sequence of images) input Code mostly copied from:
    https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/models/multi_modal/video_synthesis/unet_sd.py#L1016

    Parameters:
        in_dim (`int`): Number of input channels.
        out_dim (`int`): Number of output channels.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: Optional[int] = None,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        dtype: str = "float16",
    ):
        super().__init__()
        out_dim = out_dim or in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim

        # NOTE: using SiLU as a module avoids complicated weight mapping due to nn.Sequential ordering
        # conv layers

        kernel_size = (3, 1, 1)
        padding = [k // 2 for k in kernel_size]

        self.conv1 = nn.Sequential(
            nn.GroupNorm(norm_num_groups, in_dim, dtype=dtype),
            SiLU(),
            nn.Conv3d(in_dim, out_dim, kernel_size, padding=padding, dtype=dtype),
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(norm_num_groups, out_dim, dtype=dtype),
            SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, kernel_size, padding=padding, dtype=dtype),
        )
        self.conv3 = nn.Sequential(
            nn.GroupNorm(norm_num_groups, out_dim, dtype=dtype),
            SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, kernel_size, padding=padding, dtype=dtype),
        )
        self.conv4 = nn.Sequential(
            nn.GroupNorm(norm_num_groups, out_dim, dtype=dtype),
            SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, kernel_size, padding=padding, dtype=dtype),
        )

    def forward(self, hidden_states: Tensor, num_frames: int = 1) -> Tensor:
        shape = ops.size()(hidden_states)
        hidden_states = ops.reshape()(
            ops.unsqueeze(0)(hidden_states), shape=[-1, num_frames] + shape[1:]
        )

        identity = hidden_states
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.conv3(hidden_states)
        hidden_states = self.conv4(hidden_states)

        hidden_states = identity + hidden_states
        shape = ops.size()(hidden_states)
        hidden_states = ops.reshape()(
            hidden_states, shape=[shape[0] * shape[1], shape[2], shape[3], shape[4]]
        )
        return hidden_states


class TemporalResnetBlock(nn.Module):
    r"""
    A Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        temb_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        temb_channels: int = 512,
        eps: float = 1e-6,
        dtype: str = "float16",
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        kernel_size = (3, 1, 1)
        padding = [k // 2 for k in kernel_size]

        self.norm1 = nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=eps, affine=True, dtype=dtype
        )
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dtype=dtype,
        )

        if temb_channels is not None:
            self.time_emb_proj = nn.Linear(temb_channels, out_channels, dtype=dtype)
        else:
            self.time_emb_proj = None

        self.norm2 = nn.GroupNorm(
            num_groups=32, num_channels=out_channels, eps=eps, affine=True, dtype=dtype
        )

        self.dropout = nn.Dropout(0.0)
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dtype=dtype,
        )

        self.nonlinearity = get_activation("silu")

        self.use_in_shortcut = self.in_channels != out_channels

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv3d(
                in_channels, out_channels, kernel_size=1, dtype=dtype
            )

    def forward(self, input_tensor: Tensor, temb: Tensor) -> Tensor:
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)
        if self.time_emb_proj is not None:
            temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)
            temb = ops.unsqueeze(2)(ops.unsqueeze(2)(temb))
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = input_tensor + hidden_states

        return output_tensor


# VideoResBlock
class SpatioTemporalResBlock(nn.Module):
    r"""
    A SpatioTemporal Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        temb_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the spatial resenet.
        temporal_eps (`float`, *optional*, defaults to `eps`): The epsilon to use for the temporal resnet.
        merge_factor (`float`, *optional*, defaults to `0.5`): The merge factor to use for the temporal mixing.
        merge_strategy (`str`, *optional*, defaults to `learned_with_images`):
            The merge strategy to use for the temporal mixing.
        switch_spatial_to_temporal_mix (`bool`, *optional*, defaults to `False`):
            If `True`, switch the spatial and temporal mixing.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        temb_channels: int = 512,
        eps: float = 1e-6,
        temporal_eps: Optional[float] = None,
        merge_factor: float = 0.5,
        merge_strategy="learned_with_images",
        switch_spatial_to_temporal_mix: bool = False,
        dtype: str = "float16",
    ):
        super().__init__()

        self.spatial_res_block = ResnetBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            eps=eps,
            dtype=dtype,
        )

        self.temporal_res_block = TemporalResnetBlock(
            in_channels=out_channels if out_channels is not None else in_channels,
            out_channels=out_channels if out_channels is not None else in_channels,
            temb_channels=temb_channels,
            eps=temporal_eps if temporal_eps is not None else eps,
            dtype=dtype,
        )

        self.time_mixer = AlphaBlender(
            alpha=merge_factor,
            merge_strategy=merge_strategy,
            switch_spatial_to_temporal_mix=switch_spatial_to_temporal_mix,
        )

    def forward(
        self,
        hidden_states: Tensor,
        temb: Optional[Tensor] = None,
        image_only_indicator: Optional[Tensor] = None,
    ):
        num_frames = ops.size()(image_only_indicator, dim=-1)
        hidden_states = self.spatial_res_block(hidden_states, temb)

        batch_frames, height, width, channels = ops.size()(hidden_states)
        batch_size = batch_frames._attrs["int_var"] / num_frames._attrs["int_var"]

        hidden_states_mix = ops.reshape()(
            ops.unsqueeze(0)(hidden_states),
            [batch_size, num_frames, height, width, channels],
        )
        hidden_states = ops.reshape()(
            ops.unsqueeze(0)(hidden_states),
            [batch_size, num_frames, height, width, channels],
        )

        if temb is not None:
            temb = ops.reshape()(temb, [batch_size, num_frames, -1])

        hidden_states = self.temporal_res_block(hidden_states, temb)
        hidden_states = self.time_mixer(
            x_spatial=hidden_states_mix,
            x_temporal=hidden_states,
            image_only_indicator=image_only_indicator,
        )

        hidden_states = ops.reshape()(
            hidden_states, [batch_frames, height, width, channels]
        )
        return hidden_states


class AlphaBlender(nn.Module):
    r"""
    A module to blend spatial and temporal features.

    Parameters:
        alpha (`float`): The initial value of the blending factor.
        merge_strategy (`str`, *optional*, defaults to `learned_with_images`):
            The merge strategy to use for the temporal mixing.
        switch_spatial_to_temporal_mix (`bool`, *optional*, defaults to `False`):
            If `True`, switch the spatial and temporal mixing.
    """

    strategies = ["learned", "fixed", "learned_with_images"]

    def __init__(
        self,
        alpha: float,
        merge_strategy: str = "learned_with_images",
        switch_spatial_to_temporal_mix: bool = False,
        dtype: str = "float16",
    ):
        super().__init__()
        self.dtype = dtype
        self.merge_strategy = merge_strategy
        self.switch_spatial_to_temporal_mix = (
            switch_spatial_to_temporal_mix  # For TemporalVAE
        )

        if merge_strategy not in self.strategies:
            raise ValueError(f"merge_strategy needs to be in {self.strategies}")

        if self.merge_strategy == "fixed":
            self.mix_factor = alpha
        elif (
            self.merge_strategy == "learned"
            or self.merge_strategy == "learned_with_images"
        ):
            # originally nn.Parameter
            self.mix_factor = alpha
        else:
            raise ValueError(f"Unknown merge strategy {self.merge_strategy}")

    def get_alpha(self, image_only_indicator: Tensor, ndims: int) -> Tensor:
        if self.merge_strategy == "fixed":
            alpha = self.mix_factor

        elif self.merge_strategy == "learned":
            alpha = sigmoid(self.mix_factor)

        elif self.merge_strategy == "learned_with_images":
            if image_only_indicator is None:
                raise ValueError(
                    "Please provide image_only_indicator to use learned_with_images merge strategy"
                )
            batch = ops.size()(image_only_indicator, dim=1)
            alpha = ops.where()(
                image_only_indicator, 1.0, sigmoid(self.mix_factor), dtype="float16"
            )

            # (batch, frames, height, width, channel)
            if ndims == 5:
                alpha = ops.unsqueeze(-1)(
                    ops.unsqueeze(-1)(ops.unsqueeze(-1)(alpha))
                )  # alpha[:, None, :, None, None]
            # (batch*frames, height*width, channels)
            elif ndims == 3:
                alpha = ops.unsqueeze(-1)(
                    ops.unsqueeze(-1)(ops.reshape()(alpha, [-1]))
                )  # alpha.reshape(-1)[:, None, None]
            else:
                raise ValueError(
                    f"Unexpected ndims {ndims}. Dimensions should be 3 or 5"
                )

        else:
            raise NotImplementedError

        return alpha

    def forward(
        self,
        x_spatial: Tensor,
        x_temporal: Tensor,
        image_only_indicator: Optional[Tensor] = None,
    ) -> Tensor:
        alpha = self.get_alpha(image_only_indicator, ndims=len(ops.size()(x_spatial)))
        if isinstance(alpha, Tensor):
            alpha = ops.cast()(alpha, x_spatial.dtype())

        if self.switch_spatial_to_temporal_mix:
            alpha = 1.0 - alpha

        x = alpha * x_spatial + (1.0 - alpha) * x_temporal
        return x
