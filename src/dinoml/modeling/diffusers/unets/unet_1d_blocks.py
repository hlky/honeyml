import math
from typing import Optional, Tuple, Union

from dinoml.compiler import ops

from dinoml.frontend import nn, Tensor

from ..activations import get_activation
from ..resnet import Downsample1D, rearrange_dims, ResidualTemporalBlock1D, Upsample1D


class DownResnetBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        conv_shortcut: bool = False,
        temb_channels: int = 32,
        groups: int = 32,
        groups_out: Optional[int] = None,
        non_linearity: Optional[str] = None,
        time_embedding_norm: str = "default",
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        dtype: str = "float16",
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.add_downsample = add_downsample
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        # there will always be at least one resnet
        resnets = [
            ResidualTemporalBlock1D(
                in_channels, out_channels, embed_dim=temb_channels, dtype=dtype
            )
        ]

        for _ in range(num_layers):
            resnets.append(
                ResidualTemporalBlock1D(
                    out_channels, out_channels, embed_dim=temb_channels, dtype=dtype
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if non_linearity is None:
            self.nonlinearity = None
        else:
            self.nonlinearity = get_activation(non_linearity)

        self.downsample = None
        if add_downsample:
            self.downsample = Downsample1D(
                out_channels, use_conv=True, padding=1, dtype=dtype
            )

    def forward(self, hidden_states: Tensor, temb: Optional[Tensor] = None) -> Tensor:
        output_states = ()

        hidden_states = self.resnets[0](hidden_states, temb)
        for resnet in self.resnets[1:]:
            hidden_states = resnet(hidden_states, temb)

        output_states += (hidden_states,)

        if self.nonlinearity is not None:
            hidden_states = self.nonlinearity(hidden_states)

        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states)

        return hidden_states, output_states


class UpResnetBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        temb_channels: int = 32,
        groups: int = 32,
        groups_out: Optional[int] = None,
        non_linearity: Optional[str] = None,
        time_embedding_norm: str = "default",
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        dtype: str = "float16",
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.time_embedding_norm = time_embedding_norm
        self.add_upsample = add_upsample
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        # there will always be at least one resnet
        resnets = [
            ResidualTemporalBlock1D(
                2 * in_channels, out_channels, embed_dim=temb_channels, dtype=dtype
            )
        ]

        for _ in range(num_layers):
            resnets.append(
                ResidualTemporalBlock1D(
                    out_channels, out_channels, embed_dim=temb_channels, dtype=dtype
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if non_linearity is None:
            self.nonlinearity = None
        else:
            self.nonlinearity = get_activation(non_linearity)

        self.upsample = None
        if add_upsample:
            self.upsample = Upsample1D(
                out_channels, use_conv_transpose=True, dtype=dtype
            )

    def forward(
        self,
        hidden_states: Tensor,
        res_hidden_states_tuple: Optional[Tuple[Tensor, ...]] = None,
        temb: Optional[Tensor] = None,
    ) -> Tensor:
        if res_hidden_states_tuple is not None:
            res_hidden_states = res_hidden_states_tuple[-1]
            hidden_states = ops.concatenate()((hidden_states, res_hidden_states), dim=1)

        hidden_states = self.resnets[0](hidden_states, temb)
        for resnet in self.resnets[1:]:
            hidden_states = resnet(hidden_states, temb)

        if self.nonlinearity is not None:
            hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            hidden_states = self.upsample(hidden_states)

        return hidden_states


class ValueFunctionMidBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int,
        dtype: str = "float16",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim

        self.res1 = ResidualTemporalBlock1D(
            in_channels, in_channels // 2, embed_dim=embed_dim, dtype=dtype
        )
        self.down1 = Downsample1D(out_channels // 2, use_conv=True, dtype=dtype)
        self.res2 = ResidualTemporalBlock1D(
            in_channels // 2, in_channels // 4, embed_dim=embed_dim, dtype=dtype
        )
        self.down2 = Downsample1D(out_channels // 4, use_conv=True, dtype=dtype)

    def forward(self, x: Tensor, temb: Optional[Tensor] = None) -> Tensor:
        x = self.res1(x, temb)
        x = self.down1(x)
        x = self.res2(x, temb)
        x = self.down2(x)
        return x


class MidResTemporalBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int,
        num_layers: int = 1,
        add_downsample: bool = False,
        add_upsample: bool = False,
        non_linearity: Optional[str] = None,
        dtype: str = "float16",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_downsample = add_downsample

        # there will always be at least one resnet
        resnets = [
            ResidualTemporalBlock1D(
                in_channels, out_channels, embed_dim=embed_dim, dtype=dtype
            )
        ]

        for _ in range(num_layers):
            resnets.append(
                ResidualTemporalBlock1D(
                    out_channels, out_channels, embed_dim=embed_dim, dtype=dtype
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if non_linearity is None:
            self.nonlinearity = None
        else:
            self.nonlinearity = get_activation(non_linearity)

        self.upsample = None
        if add_upsample:
            self.upsample = Upsample1D(out_channels, use_conv=True, dtype=dtype)

        self.downsample = None
        if add_downsample:
            self.downsample = Downsample1D(out_channels, use_conv=True, dtype=dtype)

        if self.upsample and self.downsample:
            raise ValueError("Block cannot downsample and upsample")

    def forward(self, hidden_states: Tensor, temb: Tensor) -> Tensor:
        hidden_states = self.resnets[0](hidden_states, temb)
        for resnet in self.resnets[1:]:
            hidden_states = resnet(hidden_states, temb)

        if self.upsample:
            hidden_states = self.upsample(hidden_states)
        if self.downsample:
            self.downsample = self.downsample(hidden_states)

        return hidden_states


class OutConv1DBlock(nn.Module):
    def __init__(
        self,
        num_groups_out: int,
        out_channels: int,
        embed_dim: int,
        act_fn: str,
        dtype: str = "float16",
    ):
        super().__init__()
        self.final_conv1d_1 = nn.Conv1d(embed_dim, embed_dim, 5, padding=2, dtype=dtype)
        self.final_conv1d_gn = nn.GroupNorm(num_groups_out, embed_dim, dtype=dtype)
        self.final_conv1d_act = get_activation(act_fn)
        self.final_conv1d_2 = nn.Conv1d(embed_dim, out_channels, 1, dtype=dtype)

    def forward(self, hidden_states: Tensor, temb: Optional[Tensor] = None) -> Tensor:
        hidden_states = self.final_conv1d_1(hidden_states)
        hidden_states = rearrange_dims(hidden_states)
        hidden_states = self.final_conv1d_gn(hidden_states)
        hidden_states = rearrange_dims(hidden_states)
        hidden_states = self.final_conv1d_act(hidden_states)
        hidden_states = self.final_conv1d_2(hidden_states)
        return hidden_states


class OutValueFunctionBlock(nn.Module):
    def __init__(
        self,
        fc_dim: int,
        embed_dim: int,
        act_fn: str = "mish",
        dtype: str = "float16",
    ):
        super().__init__()
        self.final_block = nn.ModuleList(
            [
                nn.Linear(fc_dim + embed_dim, fc_dim // 2, dtype=dtype),
                get_activation(act_fn),
                nn.Linear(fc_dim // 2, 1, dtype=dtype),
            ]
        )

    def forward(self, hidden_states: Tensor, temb: Tensor) -> Tensor:
        hidden_states = ops.reshape()(
            hidden_states, [ops.size()(hidden_states, dim=0), -1]
        )
        hidden_states = ops.concatenate()((hidden_states, temb), dim=-1)
        for layer in self.final_block:
            hidden_states = layer(hidden_states)

        return hidden_states


_kernels = {
    "linear": [1 / 8, 3 / 8, 3 / 8, 1 / 8],
    "cubic": [
        -0.01171875,
        -0.03515625,
        0.11328125,
        0.43359375,
        0.43359375,
        0.11328125,
        -0.03515625,
        -0.01171875,
    ],
    "lanczos3": [
        0.003689131001010537,
        0.015056144446134567,
        -0.03399861603975296,
        -0.066637322306633,
        0.13550527393817902,
        0.44638532400131226,
        0.44638532400131226,
        0.13550527393817902,
        -0.066637322306633,
        -0.03399861603975296,
        0.015056144446134567,
        0.003689131001010537,
    ],
}


class Downsample1d(nn.Module):
    def __init__(self, kernel: str = "linear", pad_mode: str = "reflect"):
        super().__init__()
        raise NotImplementedError("kernel selection")
        self.pad_mode = pad_mode
        kernel_1d = torch.tensor(_kernels[kernel])
        self.pad = ops.size()(kernel_1d, dim=0)["int_var"].upper_bound() // 2 - 1
        self.kernel = kernel_1d

    def forward(self, hidden_states: Tensor) -> Tensor:
        raise NotImplementedError("weight assignment")
        hidden_states = ops.pad((self.pad,) * 2, mode=self.pad_mode)(hidden_states)
        hidden_states_dim1 = ops.size()(hidden_states, dim=1)
        kernel_dim0 = ops.size()(self.kernel, dim=0)
        weight = ops.full()(
            hidden_states_dim1,
            hidden_states_dim1,
            kernel_dim0,
            fill_value=0.0,
            dtype=hidden_states.dtype(),
        )
        indices = ops.arange(0, hidden_states_dim1["int_var"], 1)()
        kernel = ops.expand()(
            ops.unsqueeze(0)(ops.cast()(self.kernel, weight.dtype())), hidden_states, -1
        )
        # TODO
        weight[indices, indices] = kernel
        # TODO: nicer interface for conv1d
        return ops.squeeze(2)(
            ops.conv2d(stride=(2, 1), pad=(0, 0), dilate=(1, 1), bias=False)(
                ops.unsqueeze(2)(hidden_states), ops.unsqueeze(2)(weight)
            )
        )


class Upsample1d(nn.Module):
    def __init__(self, kernel: str = "linear", pad_mode: str = "reflect"):
        super().__init__()
        raise NotImplementedError("kernel selection")
        self.pad_mode = pad_mode
        kernel_1d = torch.tensor(_kernels[kernel]) * 2
        self.pad = ops.size()(kernel_1d, dim=0)["int_var"].upper_bound() // 2 - 1
        self.kernel = kernel_1d

    def forward(self, hidden_states: Tensor, temb: Optional[Tensor] = None) -> Tensor:
        hidden_states = ops.pad(((self.pad + 1) // 2,) * 2, mode=self.pad_mode)(
            hidden_states
        )
        hidden_states_dim1 = ops.size()(hidden_states, dim=1)
        kernel_dim0 = ops.size()(self.kernel, dim=0)
        weight = ops.full()(
            hidden_states_dim1,
            hidden_states_dim1,
            kernel_dim0,
            fill_value=0.0,
            dtype=hidden_states.dtype(),
        )
        indices = ops.arange(0, hidden_states_dim1["int_var"], 1)()
        kernel = ops.expand()(
            ops.unsqueeze(0)(ops.cast()(self.kernel, weight.dtype())), hidden_states, -1
        )
        # TODO
        weight[indices, indices] = kernel
        # TODO nicer interface for conv_transpose1d
        return ops.squeeze(2)(
            ops.transposed_conv2d(
                stride=(2, 1),
                pad=(self.pad * 2 + 1, 0),
                dilate=(1, 1),
                bias=False,
            )(ops.unsqueeze(2)(hidden_states), ops.unsqueeze(2)(weight))
        )


class SelfAttention1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_head: int = 1,
        dropout_rate: float = 0.0,
        dtype: str = "float16",
    ):
        super().__init__()
        self.channels = in_channels
        self.group_norm = nn.GroupNorm(1, num_channels=in_channels, dtype=dtype)
        self.num_heads = n_head

        self.query = nn.Linear(self.channels, self.channels, dtype=dtype)
        self.key = nn.Linear(self.channels, self.channels, dtype=dtype)
        self.value = nn.Linear(self.channels, self.channels, dtype=dtype)

        self.proj_attn = nn.Linear(self.channels, self.channels, bias=True, dtype=dtype)

        self.dropout = nn.Dropout(dropout_rate, inplace=True)

    def transpose_for_scores(self, projection: Tensor) -> Tensor:
        new_projection_shape = ops.size()(projection)[:-1] + (self.num_heads, -1)
        # move heads to 2nd position (B, T, H * D) -> (B, T, H, D) -> (B, H, T, D)
        new_projection = ops.permute0213()(
            ops.reshape()(projection, new_projection_shape)
        )
        return new_projection

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        batch, channel_dim, seq = ops.size()(hidden_states)

        hidden_states = self.group_norm(hidden_states)
        hidden_states = ops.permute021()(hidden_states)

        query_proj = self.query(hidden_states)
        key_proj = self.key(hidden_states)
        value_proj = self.value(hidden_states)

        query_states = self.transpose_for_scores(query_proj)
        key_states = self.transpose_for_scores(key_proj)
        value_states = self.transpose_for_scores(value_proj)

        scale = 1 / math.sqrt(math.sqrt(ops.size()(key_states, dim=-1)))  # FIXME

        attention_scores = ops.gemm_rcr()(
            query_states * scale, ops.permute()(0, 1, 3, 2)(key_states) * scale
        )  # FIXME
        attention_probs = ops.softmax()(attention_scores, dim=-1)

        # compute attention output
        hidden_states = ops.gemm_rcr()(attention_probs, value_states)

        # hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()
        new_hidden_states_shape = ops.size()(hidden_states)[:-2] + (self.channels,)
        hidden_states = ops.reshape()(hidden_states, new_hidden_states_shape)

        # compute next hidden_states
        hidden_states = self.proj_attn(hidden_states)
        hidden_states = ops.permute021()(hidden_states)
        hidden_states = self.dropout(hidden_states)

        output = hidden_states + residual

        return output


class ResConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        is_last: bool = False,
        dtype: str = "float16",
    ):
        super().__init__()
        self.is_last = is_last
        self.has_conv_skip = in_channels != out_channels

        if self.has_conv_skip:
            self.conv_skip = nn.Conv1d(
                in_channels, out_channels, 1, bias=False, dtype=dtype
            )

        self.conv_1 = nn.Conv1d(in_channels, mid_channels, 5, padding=2, dtype=dtype)
        self.group_norm_1 = nn.GroupNorm(1, mid_channels, dtype=dtype)
        self.gelu_1 = ops.gelu
        self.conv_2 = nn.Conv1d(mid_channels, out_channels, 5, padding=2, dtype=dtype)

        if not self.is_last:
            self.group_norm_2 = nn.GroupNorm(1, out_channels, dtype=dtype)
            self.gelu_2 = ops.gelu

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = (
            self.conv_skip(hidden_states) if self.has_conv_skip else hidden_states
        )

        hidden_states = self.conv_1(hidden_states)
        hidden_states = self.group_norm_1(hidden_states)
        hidden_states = self.gelu_1(hidden_states)
        hidden_states = self.conv_2(hidden_states)

        if not self.is_last:
            hidden_states = self.group_norm_2(hidden_states)
            hidden_states = self.gelu_2(hidden_states)

        output = hidden_states + residual
        return output


class UNetMidBlock1D(nn.Module):
    def __init__(
        self,
        mid_channels: int,
        in_channels: int,
        out_channels: Optional[int] = None,
        dtype: str = "float16",
    ):
        super().__init__()

        out_channels = in_channels if out_channels is None else out_channels

        # there is always at least one resnet
        self.down = Downsample1d("cubic")
        resnets = [
            ResConvBlock(in_channels, mid_channels, mid_channels, dtype=dtype),
            ResConvBlock(mid_channels, mid_channels, mid_channels, dtype=dtype),
            ResConvBlock(mid_channels, mid_channels, mid_channels, dtype=dtype),
            ResConvBlock(mid_channels, mid_channels, mid_channels, dtype=dtype),
            ResConvBlock(mid_channels, mid_channels, mid_channels, dtype=dtype),
            ResConvBlock(mid_channels, mid_channels, out_channels, dtype=dtype),
        ]
        attentions = [
            SelfAttention1d(mid_channels, mid_channels // 32, dtype=dtype),
            SelfAttention1d(mid_channels, mid_channels // 32, dtype=dtype),
            SelfAttention1d(mid_channels, mid_channels // 32, dtype=dtype),
            SelfAttention1d(mid_channels, mid_channels // 32, dtype=dtype),
            SelfAttention1d(mid_channels, mid_channels // 32, dtype=dtype),
            SelfAttention1d(out_channels, out_channels // 32, dtype=dtype),
        ]
        self.up = Upsample1d(kernel="cubic")

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states: Tensor, temb: Optional[Tensor] = None) -> Tensor:
        hidden_states = self.down(hidden_states)
        for attn, resnet in zip(self.attentions, self.resnets):
            hidden_states = resnet(hidden_states)
            hidden_states = attn(hidden_states)

        hidden_states = self.up(hidden_states)

        return hidden_states


class AttnDownBlock1D(nn.Module):
    def __init__(
        self,
        out_channels: int,
        in_channels: int,
        mid_channels: Optional[int] = None,
        dtype: str = "float16",
    ):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        self.down = Downsample1d("cubic")
        resnets = [
            ResConvBlock(in_channels, mid_channels, mid_channels, dtype=dtype),
            ResConvBlock(mid_channels, mid_channels, mid_channels, dtype=dtype),
            ResConvBlock(mid_channels, mid_channels, out_channels, dtype=dtype),
        ]
        attentions = [
            SelfAttention1d(mid_channels, mid_channels // 32, dtype=dtype),
            SelfAttention1d(mid_channels, mid_channels // 32, dtype=dtype),
            SelfAttention1d(out_channels, out_channels // 32, dtype=dtype),
        ]

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states: Tensor, temb: Optional[Tensor] = None) -> Tensor:
        hidden_states = self.down(hidden_states)

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states)
            hidden_states = attn(hidden_states)

        return hidden_states, (hidden_states,)


class DownBlock1D(nn.Module):
    def __init__(
        self,
        out_channels: int,
        in_channels: int,
        mid_channels: Optional[int] = None,
        dtype: str = "float16",
    ):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        self.down = Downsample1d("cubic")
        resnets = [
            ResConvBlock(in_channels, mid_channels, mid_channels, dtype=dtype),
            ResConvBlock(mid_channels, mid_channels, mid_channels, dtype=dtype),
            ResConvBlock(mid_channels, mid_channels, out_channels, dtype=dtype),
        ]

        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states: Tensor, temb: Optional[Tensor] = None) -> Tensor:
        hidden_states = self.down(hidden_states)

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        return hidden_states, (hidden_states,)


class DownBlock1DNoSkip(nn.Module):
    def __init__(
        self,
        out_channels: int,
        in_channels: int,
        mid_channels: Optional[int] = None,
        dtype: str = "float16",
    ):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        resnets = [
            ResConvBlock(in_channels, mid_channels, mid_channels, dtype=dtype),
            ResConvBlock(mid_channels, mid_channels, mid_channels, dtype=dtype),
            ResConvBlock(mid_channels, mid_channels, out_channels, dtype=dtype),
        ]

        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states: Tensor, temb: Optional[Tensor] = None) -> Tensor:
        hidden_states = ops.concatenate()([hidden_states, temb], dim=1)
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        return hidden_states, (hidden_states,)


class AttnUpBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        dtype: str = "float16",
    ):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        resnets = [
            ResConvBlock(2 * in_channels, mid_channels, mid_channels, dtype=dtype),
            ResConvBlock(mid_channels, mid_channels, mid_channels, dtype=dtype),
            ResConvBlock(mid_channels, mid_channels, out_channels, dtype=dtype),
        ]
        attentions = [
            SelfAttention1d(mid_channels, mid_channels // 32, dtype=dtype),
            SelfAttention1d(mid_channels, mid_channels // 32, dtype=dtype),
            SelfAttention1d(out_channels, out_channels // 32, dtype=dtype),
        ]

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.up = Upsample1d(kernel="cubic")

    def forward(
        self,
        hidden_states: Tensor,
        res_hidden_states_tuple: Tuple[Tensor, ...],
        temb: Optional[Tensor] = None,
    ) -> Tensor:
        res_hidden_states = res_hidden_states_tuple[-1]
        hidden_states = ops.concatenate()([hidden_states, res_hidden_states], dim=1)

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states)
            hidden_states = attn(hidden_states)

        hidden_states = self.up(hidden_states)

        return hidden_states


class UpBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        dtype: str = "float16",
    ):
        super().__init__()
        mid_channels = in_channels if mid_channels is None else mid_channels

        resnets = [
            ResConvBlock(2 * in_channels, mid_channels, mid_channels, dtype=dtype),
            ResConvBlock(mid_channels, mid_channels, mid_channels, dtype=dtype),
            ResConvBlock(mid_channels, mid_channels, out_channels, dtype=dtype),
        ]

        self.resnets = nn.ModuleList(resnets)
        self.up = Upsample1d(kernel="cubic")

    def forward(
        self,
        hidden_states: Tensor,
        res_hidden_states_tuple: Tuple[Tensor, ...],
        temb: Optional[Tensor] = None,
    ) -> Tensor:
        res_hidden_states = res_hidden_states_tuple[-1]
        hidden_states = ops.concatenate()([hidden_states, res_hidden_states], dim=1)

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        hidden_states = self.up(hidden_states)

        return hidden_states


class UpBlock1DNoSkip(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        dtype: str = "float16",
    ):
        super().__init__()
        mid_channels = in_channels if mid_channels is None else mid_channels

        resnets = [
            ResConvBlock(2 * in_channels, mid_channels, mid_channels, dtype=dtype),
            ResConvBlock(mid_channels, mid_channels, mid_channels, dtype=dtype),
            ResConvBlock(
                mid_channels, mid_channels, out_channels, is_last=True, dtype=dtype
            ),
        ]

        self.resnets = nn.ModuleList(resnets)

    def forward(
        self,
        hidden_states: Tensor,
        res_hidden_states_tuple: Tuple[Tensor, ...],
        temb: Optional[Tensor] = None,
    ) -> Tensor:
        res_hidden_states = res_hidden_states_tuple[-1]
        hidden_states = ops.concatenate()([hidden_states, res_hidden_states], dim=1)

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        return hidden_states


DownBlockType = Union[
    DownResnetBlock1D, DownBlock1D, AttnDownBlock1D, DownBlock1DNoSkip
]
MidBlockType = Union[MidResTemporalBlock1D, ValueFunctionMidBlock1D, UNetMidBlock1D]
OutBlockType = Union[OutConv1DBlock, OutValueFunctionBlock]
UpBlockType = Union[UpResnetBlock1D, UpBlock1D, AttnUpBlock1D, UpBlock1DNoSkip]


def get_down_block(
    down_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    temb_channels: int,
    add_downsample: bool,
    dtype: str = "float16",
) -> DownBlockType:
    if down_block_type == "DownResnetBlock1D":
        return DownResnetBlock1D(
            in_channels=in_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            dtype=dtype,
        )
    elif down_block_type == "DownBlock1D":
        return DownBlock1D(
            out_channels=out_channels, in_channels=in_channels, dtype=dtype
        )
    elif down_block_type == "AttnDownBlock1D":
        return AttnDownBlock1D(
            out_channels=out_channels, in_channels=in_channels, dtype=dtype
        )
    elif down_block_type == "DownBlock1DNoSkip":
        return DownBlock1DNoSkip(
            out_channels=out_channels, in_channels=in_channels, dtype=dtype
        )
    raise ValueError(f"{down_block_type} does not exist.")


def get_up_block(
    up_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    temb_channels: int,
    add_upsample: bool,
    dtype: str = "float16",
) -> UpBlockType:
    if up_block_type == "UpResnetBlock1D":
        return UpResnetBlock1D(
            in_channels=in_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            dtype=dtype,
        )
    elif up_block_type == "UpBlock1D":
        return UpBlock1D(
            in_channels=in_channels, out_channels=out_channels, dtype=dtype
        )
    elif up_block_type == "AttnUpBlock1D":
        return AttnUpBlock1D(
            in_channels=in_channels, out_channels=out_channels, dtype=dtype
        )
    elif up_block_type == "UpBlock1DNoSkip":
        return UpBlock1DNoSkip(
            in_channels=in_channels, out_channels=out_channels, dtype=dtype
        )
    raise ValueError(f"{up_block_type} does not exist.")


def get_mid_block(
    mid_block_type: str,
    num_layers: int,
    in_channels: int,
    mid_channels: int,
    out_channels: int,
    embed_dim: int,
    add_downsample: bool,
    dtype: str = "float16",
) -> MidBlockType:
    if mid_block_type == "MidResTemporalBlock1D":
        return MidResTemporalBlock1D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            embed_dim=embed_dim,
            add_downsample=add_downsample,
            dtype=dtype,
        )
    elif mid_block_type == "ValueFunctionMidBlock1D":
        return ValueFunctionMidBlock1D(
            in_channels=in_channels,
            out_channels=out_channels,
            embed_dim=embed_dim,
            dtype=dtype,
        )
    elif mid_block_type == "UNetMidBlock1D":
        return UNetMidBlock1D(
            in_channels=in_channels,
            mid_channels=mid_channels,
            out_channels=out_channels,
            dtype=dtype,
        )
    raise ValueError(f"{mid_block_type} does not exist.")


def get_out_block(
    *,
    out_block_type: str,
    num_groups_out: int,
    embed_dim: int,
    out_channels: int,
    act_fn: str,
    fc_dim: int,
    dtype: str = "float16",
) -> Optional[OutBlockType]:
    if out_block_type == "OutConv1DBlock":
        return OutConv1DBlock(
            num_groups_out, out_channels, embed_dim, act_fn, dtype=dtype
        )
    elif out_block_type == "ValueFunction":
        return OutValueFunctionBlock(fc_dim, embed_dim, act_fn, dtype=dtype)
    return None
