from typing import Any, Dict, Optional, Tuple, Union

from honey.compiler import ops

from honey.frontend import nn, Tensor

from ..attention import Attention
from ..resnet import (
    Downsample2D,
    ResnetBlock2D,
    SpatioTemporalResBlock,
    TemporalConvLayer,
    Upsample2D,
)
from ..transformers.dual_transformer_2d import DualTransformer2DModel
from ..transformers.transformer_2d import Transformer2DModel
from ..transformers.transformer_temporal import (
    TransformerSpatioTemporalModel,
    TransformerTemporalModel,
)


def get_down_block(
    down_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    temb_channels: int,
    add_downsample: bool,
    resnet_eps: float,
    resnet_act_fn: str,
    num_attention_heads: int,
    resnet_groups: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    downsample_padding: Optional[int] = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = True,
    only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default",
    temporal_num_attention_heads: int = 8,
    temporal_max_seq_length: int = 32,
    transformer_layers_per_block: int = 1,
    dtype: str = "float16",
) -> Union[
    "DownBlock3D",
    "CrossAttnDownBlock3D",
    "DownBlockMotion",
    "CrossAttnDownBlockMotion",
    "DownBlockSpatioTemporal",
    "CrossAttnDownBlockSpatioTemporal",
]:
    if down_block_type == "DownBlock3D":
        return DownBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
            dtype=dtype,
        )
    elif down_block_type == "CrossAttnDownBlock3D":
        if cross_attention_dim is None:
            raise ValueError(
                "cross_attention_dim must be specified for CrossAttnDownBlock3D"
            )
        return CrossAttnDownBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            dtype=dtype,
        )
    if down_block_type == "DownBlockMotion":
        return DownBlockMotion(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
            temporal_num_attention_heads=temporal_num_attention_heads,
            temporal_max_seq_length=temporal_max_seq_length,
            dtype=dtype,
        )
    elif down_block_type == "CrossAttnDownBlockMotion":
        if cross_attention_dim is None:
            raise ValueError(
                "cross_attention_dim must be specified for CrossAttnDownBlockMotion"
            )
        return CrossAttnDownBlockMotion(
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            temporal_num_attention_heads=temporal_num_attention_heads,
            temporal_max_seq_length=temporal_max_seq_length,
            dtype=dtype,
        )
    elif down_block_type == "DownBlockSpatioTemporal":
        # added for SDV
        return DownBlockSpatioTemporal(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            dtype=dtype,
        )
    elif down_block_type == "CrossAttnDownBlockSpatioTemporal":
        # added for SDV
        if cross_attention_dim is None:
            raise ValueError(
                "cross_attention_dim must be specified for CrossAttnDownBlockSpatioTemporal"
            )
        return CrossAttnDownBlockSpatioTemporal(
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            add_downsample=add_downsample,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            dtype=dtype,
        )

    raise ValueError(f"{down_block_type} does not exist.")


def get_up_block(
    up_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    prev_output_channel: int,
    temb_channels: int,
    add_upsample: bool,
    resnet_eps: float,
    resnet_act_fn: str,
    num_attention_heads: int,
    resolution_idx: Optional[int] = None,
    resnet_groups: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = True,
    only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default",
    temporal_num_attention_heads: int = 8,
    temporal_cross_attention_dim: Optional[int] = None,
    temporal_max_seq_length: int = 32,
    transformer_layers_per_block: int = 1,
    dropout: float = 0.0,
    dtype: str = "float16",
) -> Union[
    "UpBlock3D",
    "CrossAttnUpBlock3D",
    "UpBlockMotion",
    "CrossAttnUpBlockMotion",
    "UpBlockSpatioTemporal",
    "CrossAttnUpBlockSpatioTemporal",
]:
    if up_block_type == "UpBlock3D":
        return UpBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resolution_idx=resolution_idx,
            dtype=dtype,
        )
    elif up_block_type == "CrossAttnUpBlock3D":
        if cross_attention_dim is None:
            raise ValueError(
                "cross_attention_dim must be specified for CrossAttnUpBlock3D"
            )
        return CrossAttnUpBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resolution_idx=resolution_idx,
            dtype=dtype,
        )
    if up_block_type == "UpBlockMotion":
        return UpBlockMotion(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resolution_idx=resolution_idx,
            temporal_num_attention_heads=temporal_num_attention_heads,
            temporal_max_seq_length=temporal_max_seq_length,
            dtype=dtype,
        )
    elif up_block_type == "CrossAttnUpBlockMotion":
        if cross_attention_dim is None:
            raise ValueError(
                "cross_attention_dim must be specified for CrossAttnUpBlockMotion"
            )
        return CrossAttnUpBlockMotion(
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resolution_idx=resolution_idx,
            temporal_num_attention_heads=temporal_num_attention_heads,
            temporal_max_seq_length=temporal_max_seq_length,
            dtype=dtype,
        )
    elif up_block_type == "UpBlockSpatioTemporal":
        # added for SDV
        return UpBlockSpatioTemporal(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            resolution_idx=resolution_idx,
            add_upsample=add_upsample,
            dtype=dtype,
        )
    elif up_block_type == "CrossAttnUpBlockSpatioTemporal":
        # added for SDV
        if cross_attention_dim is None:
            raise ValueError(
                "cross_attention_dim must be specified for CrossAttnUpBlockSpatioTemporal"
            )
        return CrossAttnUpBlockSpatioTemporal(
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            add_upsample=add_upsample,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            resolution_idx=resolution_idx,
            dtype=dtype,
        )

    raise ValueError(f"{up_block_type} does not exist.")


class UNetMidBlock3DCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 1,
        output_scale_factor: float = 1.0,
        cross_attention_dim: int = 1280,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = True,
        upcast_attention: bool = False,
        dtype: str = "float16",
    ):
        super().__init__()

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        resnet_groups = (
            resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        )

        # there is always at least one resnet
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
                dtype=dtype,
            )
        ]
        temp_convs = [
            TemporalConvLayer(
                in_channels,
                in_channels,
                dropout=0.1,
                norm_num_groups=resnet_groups,
                dtype=dtype,
            )
        ]
        attentions = []
        temp_attentions = []

        for _ in range(num_layers):
            attentions.append(
                Transformer2DModel(
                    in_channels // num_attention_heads,
                    num_attention_heads,
                    in_channels=in_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    upcast_attention=upcast_attention,
                    dtype=dtype,
                )
            )
            temp_attentions.append(
                TransformerTemporalModel(
                    in_channels // num_attention_heads,
                    num_attention_heads,
                    in_channels=in_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    dtype=dtype,
                )
            )
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    dtype=dtype,
                )
            )
            temp_convs.append(
                TemporalConvLayer(
                    in_channels,
                    in_channels,
                    dropout=0.1,
                    norm_num_groups=resnet_groups,
                    dtype=dtype,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)
        self.temp_attentions = nn.ModuleList(temp_attentions)

    def forward(
        self,
        hidden_states: Tensor,
        temb: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        num_frames: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        hidden_states = self.resnets[0](hidden_states, temb)
        hidden_states = self.temp_convs[0](hidden_states, num_frames=num_frames)
        for attn, temp_attn, resnet, temp_conv in zip(
            self.attentions, self.temp_attentions, self.resnets[1:], self.temp_convs[1:]
        ):
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]
            hidden_states = temp_attn(
                hidden_states,
                num_frames=num_frames,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]
            hidden_states = resnet(hidden_states, temb)
            hidden_states = temp_conv(hidden_states, num_frames=num_frames)

        return hidden_states


class CrossAttnDownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
        output_scale_factor: float = 1.0,
        downsample_padding: int = 1,
        add_downsample: bool = True,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        dtype: str = "float16",
    ):
        super().__init__()
        resnets = []
        attentions = []
        temp_attentions = []
        temp_convs = []

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    dtype=dtype,
                )
            )
            temp_convs.append(
                TemporalConvLayer(
                    out_channels,
                    out_channels,
                    dropout=0.1,
                    norm_num_groups=resnet_groups,
                    dtype=dtype,
                )
            )
            attentions.append(
                Transformer2DModel(
                    out_channels // num_attention_heads,
                    num_attention_heads,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    dtype=dtype,
                )
            )
            temp_attentions.append(
                TransformerTemporalModel(
                    out_channels // num_attention_heads,
                    num_attention_heads,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    dtype=dtype,
                )
            )
        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)
        self.temp_attentions = nn.ModuleList(temp_attentions)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                        dtype=dtype,
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: Tensor,
        temb: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        num_frames: int = 1,
        cross_attention_kwargs: Dict[str, Any] = None,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        # TODO(Patrick, William) - attention mask is not used
        output_states = ()

        for resnet, temp_conv, attn, temp_attn in zip(
            self.resnets, self.temp_convs, self.attentions, self.temp_attentions
        ):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = temp_conv(hidden_states, num_frames=num_frames)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]
            hidden_states = temp_attn(
                hidden_states,
                num_frames=num_frames,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class DownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
        dtype: str = "float16",
    ):
        super().__init__()
        resnets = []
        temp_convs = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    dtype=dtype,
                )
            )
            temp_convs.append(
                TemporalConvLayer(
                    out_channels,
                    out_channels,
                    dropout=0.1,
                    norm_num_groups=resnet_groups,
                    dtype=dtype,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                        dtype=dtype,
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: Tensor,
        temb: Optional[Tensor] = None,
        num_frames: int = 1,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        output_states = ()

        for resnet, temp_conv in zip(self.resnets, self.temp_convs):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = temp_conv(hidden_states, num_frames=num_frames)

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class CrossAttnUpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        resolution_idx: Optional[int] = None,
        dtype: str = "float16",
    ):
        super().__init__()
        resnets = []
        temp_convs = []
        attentions = []
        temp_attentions = []

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    dtype=dtype,
                )
            )
            temp_convs.append(
                TemporalConvLayer(
                    out_channels,
                    out_channels,
                    dropout=0.1,
                    norm_num_groups=resnet_groups,
                    dtype=dtype,
                )
            )
            attentions.append(
                Transformer2DModel(
                    out_channels // num_attention_heads,
                    num_attention_heads,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    dtype=dtype,
                )
            )
            temp_attentions.append(
                TransformerTemporalModel(
                    out_channels // num_attention_heads,
                    num_attention_heads,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    dtype=dtype,
                )
            )
        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)
        self.temp_attentions = nn.ModuleList(temp_attentions)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [
                    Upsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        dtype=dtype,
                    )
                ]
            )
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx

    def forward(
        self,
        hidden_states: Tensor,
        res_hidden_states_tuple: Tuple[Tensor, ...],
        temb: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        upsample_size: Optional[int] = None,
        attention_mask: Optional[Tensor] = None,
        num_frames: int = 1,
        cross_attention_kwargs: Dict[str, Any] = None,
    ) -> Tensor:
        # is_freeu_enabled = (
        #     getattr(self, "s1", None)
        #     and getattr(self, "s2", None)
        #     and getattr(self, "b1", None)
        #     and getattr(self, "b2", None)
        # )

        # TODO(Patrick, William) - attention mask is not used
        for resnet, temp_conv, attn, temp_attn in zip(
            self.resnets, self.temp_convs, self.attentions, self.temp_attentions
        ):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            # # FreeU: Only operate on the first two stages
            # if is_freeu_enabled:
            #     hidden_states, res_hidden_states = apply_freeu(
            #         self.resolution_idx,
            #         hidden_states,
            #         res_hidden_states,
            #         s1=self.s1,
            #         s2=self.s2,
            #         b1=self.b1,
            #         b2=self.b2,
            #     )

            hidden_states = ops.concatenate()(
                [hidden_states, res_hidden_states], dim=-1
            )

            hidden_states = resnet(hidden_states, temb)
            hidden_states = temp_conv(hidden_states, num_frames=num_frames)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]
            hidden_states = temp_attn(
                hidden_states,
                num_frames=num_frames,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


class UpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        resolution_idx: Optional[int] = None,
        dtype: str = "float16",
    ):
        super().__init__()
        resnets = []
        temp_convs = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    dtype=dtype,
                )
            )
            temp_convs.append(
                TemporalConvLayer(
                    out_channels,
                    out_channels,
                    dropout=0.1,
                    norm_num_groups=resnet_groups,
                    dtype=dtype,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [
                    Upsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        dtype=dtype,
                    )
                ]
            )
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx

    def forward(
        self,
        hidden_states: Tensor,
        res_hidden_states_tuple: Tuple[Tensor, ...],
        temb: Optional[Tensor] = None,
        upsample_size: Optional[int] = None,
        num_frames: int = 1,
    ) -> Tensor:
        # is_freeu_enabled = (
        #     getattr(self, "s1", None)
        #     and getattr(self, "s2", None)
        #     and getattr(self, "b1", None)
        #     and getattr(self, "b2", None)
        # )
        for resnet, temp_conv in zip(self.resnets, self.temp_convs):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            # # FreeU: Only operate on the first two stages
            # if is_freeu_enabled:
            #     hidden_states, res_hidden_states = apply_freeu(
            #         self.resolution_idx,
            #         hidden_states,
            #         res_hidden_states,
            #         s1=self.s1,
            #         s2=self.s2,
            #         b1=self.b1,
            #         b2=self.b2,
            #     )

            hidden_states = ops.concatenate()(
                [hidden_states, res_hidden_states], dim=-1
            )

            hidden_states = resnet(hidden_states, temb)
            hidden_states = temp_conv(hidden_states, num_frames=num_frames)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


class DownBlockMotion(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
        temporal_num_attention_heads: int = 1,
        temporal_cross_attention_dim: Optional[int] = None,
        temporal_max_seq_length: int = 32,
        dtype: str = "float16",
    ):
        super().__init__()
        resnets = []
        motion_modules = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    dtype=dtype,
                )
            )
            motion_modules.append(
                TransformerTemporalModel(
                    num_attention_heads=temporal_num_attention_heads,
                    in_channels=out_channels,
                    norm_num_groups=resnet_groups,
                    cross_attention_dim=temporal_cross_attention_dim,
                    attention_bias=False,
                    activation_fn="geglu",
                    positional_embeddings="sinusoidal",
                    num_positional_embeddings=temporal_max_seq_length,
                    attention_head_dim=out_channels // temporal_num_attention_heads,
                    dtype=dtype,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.motion_modules = nn.ModuleList(motion_modules)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                        dtype=dtype,
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: Tensor,
        temb: Optional[Tensor] = None,
        num_frames: int = 1,
        *args,
        **kwargs,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        output_states = ()

        blocks = zip(self.resnets, self.motion_modules)
        for resnet, motion_module in blocks:
            hidden_states = resnet(hidden_states, temb)
            hidden_states = motion_module(hidden_states, num_frames=num_frames)[0]

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class CrossAttnDownBlockMotion(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
        output_scale_factor: float = 1.0,
        downsample_padding: int = 1,
        add_downsample: bool = True,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        attention_type: str = "default",
        temporal_cross_attention_dim: Optional[int] = None,
        temporal_num_attention_heads: int = 8,
        temporal_max_seq_length: int = 32,
        dtype: str = "float16",
    ):
        super().__init__()
        resnets = []
        attentions = []
        motion_modules = []

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    dtype=dtype,
                )
            )

            if not dual_cross_attention:
                attentions.append(
                    Transformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        attention_type=attention_type,
                        dtype=dtype,
                    )
                )
            else:
                attentions.append(
                    DualTransformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        dtype=dtype,
                    )
                )

            motion_modules.append(
                TransformerTemporalModel(
                    num_attention_heads=temporal_num_attention_heads,
                    in_channels=out_channels,
                    norm_num_groups=resnet_groups,
                    cross_attention_dim=temporal_cross_attention_dim,
                    attention_bias=False,
                    activation_fn="geglu",
                    positional_embeddings="sinusoidal",
                    num_positional_embeddings=temporal_max_seq_length,
                    attention_head_dim=out_channels // temporal_num_attention_heads,
                    dtype=dtype,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.motion_modules = nn.ModuleList(motion_modules)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                        dtype=dtype,
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: Tensor,
        temb: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        num_frames: int = 1,
        encoder_attention_mask: Optional[Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        additional_residuals: Optional[Tensor] = None,
    ):
        output_states = ()

        blocks = list(zip(self.resnets, self.attentions, self.motion_modules))
        for i, (resnet, attn, motion_module) in enumerate(blocks):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]
            hidden_states = motion_module(
                hidden_states,
                num_frames=num_frames,
            )[0]

            # apply additional residuals to the output of the last pair of resnet and attention blocks
            if i == len(blocks) - 1 and additional_residuals is not None:
                hidden_states = hidden_states + additional_residuals

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class CrossAttnUpBlockMotion(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        resolution_idx: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        attention_type: str = "default",
        temporal_cross_attention_dim: Optional[int] = None,
        temporal_num_attention_heads: int = 8,
        temporal_max_seq_length: int = 32,
        dtype: str = "float16",
    ):
        super().__init__()
        resnets = []
        attentions = []
        motion_modules = []

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    dtype=dtype,
                )
            )

            if not dual_cross_attention:
                attentions.append(
                    Transformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        attention_type=attention_type,
                        dtype=dtype,
                    )
                )
            else:
                attentions.append(
                    DualTransformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        dtype=dtype,
                    )
                )
            motion_modules.append(
                TransformerTemporalModel(
                    num_attention_heads=temporal_num_attention_heads,
                    in_channels=out_channels,
                    norm_num_groups=resnet_groups,
                    cross_attention_dim=temporal_cross_attention_dim,
                    attention_bias=False,
                    activation_fn="geglu",
                    positional_embeddings="sinusoidal",
                    num_positional_embeddings=temporal_max_seq_length,
                    attention_head_dim=out_channels // temporal_num_attention_heads,
                    dtype=dtype,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.motion_modules = nn.ModuleList(motion_modules)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [
                    Upsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        dtype=dtype,
                    )
                ]
            )
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx

    def forward(
        self,
        hidden_states: Tensor,
        res_hidden_states_tuple: Tuple[Tensor, ...],
        temb: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        upsample_size: Optional[int] = None,
        attention_mask: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        num_frames: int = 1,
    ) -> Tensor:
        # is_freeu_enabled = (
        #     getattr(self, "s1", None)
        #     and getattr(self, "s2", None)
        #     and getattr(self, "b1", None)
        #     and getattr(self, "b2", None)
        # )

        blocks = zip(self.resnets, self.attentions, self.motion_modules)
        for resnet, attn, motion_module in blocks:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            # # FreeU: Only operate on the first two stages
            # if is_freeu_enabled:
            #     hidden_states, res_hidden_states = apply_freeu(
            #         self.resolution_idx,
            #         hidden_states,
            #         res_hidden_states,
            #         s1=self.s1,
            #         s2=self.s2,
            #         b1=self.b1,
            #         b2=self.b2,
            #     )

            hidden_states = ops.concatenate()(
                [hidden_states, res_hidden_states], dim=-1
            )

            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]
            hidden_states = motion_module(
                hidden_states,
                num_frames=num_frames,
            )[0]

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


class UpBlockMotion(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        resolution_idx: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        temporal_norm_num_groups: int = 32,
        temporal_cross_attention_dim: Optional[int] = None,
        temporal_num_attention_heads: int = 8,
        temporal_max_seq_length: int = 32,
        dtype: str = "float16",
    ):
        super().__init__()
        resnets = []
        motion_modules = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    dtype=dtype,
                )
            )

            motion_modules.append(
                TransformerTemporalModel(
                    num_attention_heads=temporal_num_attention_heads,
                    in_channels=out_channels,
                    norm_num_groups=temporal_norm_num_groups,
                    cross_attention_dim=temporal_cross_attention_dim,
                    attention_bias=False,
                    activation_fn="geglu",
                    positional_embeddings="sinusoidal",
                    num_positional_embeddings=temporal_max_seq_length,
                    attention_head_dim=out_channels // temporal_num_attention_heads,
                    dtype=dtype,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.motion_modules = nn.ModuleList(motion_modules)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [
                    Upsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        dtype=dtype,
                    )
                ]
            )
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx

    def forward(
        self,
        hidden_states: Tensor,
        res_hidden_states_tuple: Tuple[Tensor, ...],
        temb: Optional[Tensor] = None,
        upsample_size=None,
        num_frames: int = 1,
        *args,
        **kwargs,
    ) -> Tensor:
        # is_freeu_enabled = (
        #     getattr(self, "s1", None)
        #     and getattr(self, "s2", None)
        #     and getattr(self, "b1", None)
        #     and getattr(self, "b2", None)
        # )

        blocks = zip(self.resnets, self.motion_modules)

        for resnet, motion_module in blocks:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            # # FreeU: Only operate on the first two stages
            # if is_freeu_enabled:
            #     hidden_states, res_hidden_states = apply_freeu(
            #         self.resolution_idx,
            #         hidden_states,
            #         res_hidden_states,
            #         s1=self.s1,
            #         s2=self.s2,
            #         b1=self.b1,
            #         b2=self.b2,
            #     )

            hidden_states = ops.concatenate()(
                [hidden_states, res_hidden_states], dim=-1
            )

            hidden_states = resnet(hidden_states, temb)
            hidden_states = motion_module(hidden_states, num_frames=num_frames)[0]

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


class UNetMidBlockCrossAttnMotion(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads: int = 1,
        output_scale_factor: float = 1.0,
        cross_attention_dim: int = 1280,
        dual_cross_attention: float = False,
        use_linear_projection: float = False,
        upcast_attention: float = False,
        attention_type: str = "default",
        temporal_num_attention_heads: int = 1,
        temporal_cross_attention_dim: Optional[int] = None,
        temporal_max_seq_length: int = 32,
        dtype: str = "float16",
    ):
        super().__init__()

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        resnet_groups = (
            resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        )

        # there is always at least one resnet
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
                dtype=dtype,
            )
        ]
        attentions = []
        motion_modules = []

        for _ in range(num_layers):
            if not dual_cross_attention:
                attentions.append(
                    Transformer2DModel(
                        num_attention_heads,
                        in_channels // num_attention_heads,
                        in_channels=in_channels,
                        num_layers=transformer_layers_per_block,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        upcast_attention=upcast_attention,
                        attention_type=attention_type,
                        dtype=dtype,
                    )
                )
            else:
                attentions.append(
                    DualTransformer2DModel(
                        num_attention_heads,
                        in_channels // num_attention_heads,
                        in_channels=in_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        dtype=dtype,
                    )
                )
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    dtype=dtype,
                )
            )
            motion_modules.append(
                TransformerTemporalModel(
                    num_attention_heads=temporal_num_attention_heads,
                    attention_head_dim=in_channels // temporal_num_attention_heads,
                    in_channels=in_channels,
                    norm_num_groups=resnet_groups,
                    cross_attention_dim=temporal_cross_attention_dim,
                    attention_bias=False,
                    positional_embeddings="sinusoidal",
                    num_positional_embeddings=temporal_max_seq_length,
                    activation_fn="geglu",
                    dtype=dtype,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.motion_modules = nn.ModuleList(motion_modules)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: Tensor,
        temb: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        num_frames: int = 1,
    ) -> Tensor:
        hidden_states = self.resnets[0](hidden_states, temb)

        blocks = zip(self.attentions, self.resnets[1:], self.motion_modules)
        for attn, resnet, motion_module in blocks:
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]
            hidden_states = motion_module(
                hidden_states,
                num_frames=num_frames,
            )[0]
            hidden_states = resnet(hidden_states, temb)

        return hidden_states


class MidBlockTemporalDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        attention_head_dim: int = 512,
        num_layers: int = 1,
        upcast_attention: bool = False,
        dtype: str = "float16",
    ):
        super().__init__()

        resnets = []
        attentions = []
        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels
            resnets.append(
                SpatioTemporalResBlock(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=1e-6,
                    temporal_eps=1e-5,
                    merge_factor=0.0,
                    merge_strategy="learned",
                    switch_spatial_to_temporal_mix=True,
                    dtype=dtype,
                )
            )

        attentions.append(
            Attention(
                query_dim=in_channels,
                heads=in_channels // attention_head_dim,
                dim_head=attention_head_dim,
                eps=1e-6,
                upcast_attention=upcast_attention,
                norm_num_groups=32,
                bias=True,
                residual_connection=True,
                dtype=dtype,
            )
        )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(
        self,
        hidden_states: Tensor,
        image_only_indicator: Tensor,
    ):
        hidden_states = self.resnets[0](
            hidden_states,
            image_only_indicator=image_only_indicator,
        )
        for resnet, attn in zip(self.resnets[1:], self.attentions):
            hidden_states = attn(hidden_states)
            hidden_states = resnet(
                hidden_states,
                image_only_indicator=image_only_indicator,
            )

        return hidden_states


class UpBlockTemporalDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        add_upsample: bool = True,
        dtype: str = "float16",
    ):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                SpatioTemporalResBlock(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=1e-6,
                    temporal_eps=1e-5,
                    merge_factor=0.0,
                    merge_strategy="learned",
                    switch_spatial_to_temporal_mix=True,
                    dtype=dtype,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [
                    Upsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        dtype=dtype,
                    )
                ]
            )
        else:
            self.upsamplers = None

    def forward(
        self,
        hidden_states: Tensor,
        image_only_indicator: Tensor,
    ) -> Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(
                hidden_states,
                image_only_indicator=image_only_indicator,
            )

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class UNetMidBlockSpatioTemporal(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        num_layers: int = 1,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
        dtype: str = "float16",
    ):
        super().__init__()

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads

        # support for variable transformer layers per block
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        # there is always at least one resnet
        resnets = [
            SpatioTemporalResBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=1e-5,
                dtype=dtype,
            )
        ]
        attentions = []

        for i in range(num_layers):
            attentions.append(
                TransformerSpatioTemporalModel(
                    num_attention_heads,
                    in_channels // num_attention_heads,
                    in_channels=in_channels,
                    num_layers=transformer_layers_per_block[i],
                    cross_attention_dim=cross_attention_dim,
                    dtype=dtype,
                )
            )

            resnets.append(
                SpatioTemporalResBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=1e-5,
                    dtype=dtype,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: Tensor,
        temb: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        image_only_indicator: Optional[Tensor] = None,
    ) -> Tensor:
        hidden_states = self.resnets[0](
            hidden_states,
            temb,
            image_only_indicator=image_only_indicator,
        )

        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                image_only_indicator=image_only_indicator,
                return_dict=False,
            )[0]
            hidden_states = resnet(
                hidden_states,
                temb,
                image_only_indicator=image_only_indicator,
            )

        return hidden_states


class DownBlockSpatioTemporal(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int = 1,
        add_downsample: bool = True,
        dtype: str = "float16",
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                SpatioTemporalResBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=1e-5,
                    dtype=dtype,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        name="op",
                        dtype=dtype,
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: Tensor,
        temb: Optional[Tensor] = None,
        image_only_indicator: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        output_states = ()
        for resnet in self.resnets:
            hidden_states = resnet(
                hidden_states,
                temb,
                image_only_indicator=image_only_indicator,
            )

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class CrossAttnDownBlockSpatioTemporal(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int = 1,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
        add_downsample: bool = True,
        dtype: str = "float16",
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                SpatioTemporalResBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=1e-6,
                    dtype=dtype,
                )
            )
            attentions.append(
                TransformerSpatioTemporalModel(
                    num_attention_heads,
                    out_channels // num_attention_heads,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block[i],
                    cross_attention_dim=cross_attention_dim,
                    dtype=dtype,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=1,
                        name="op",
                        dtype=dtype,
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: Tensor,
        temb: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        image_only_indicator: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        output_states = ()

        blocks = list(zip(self.resnets, self.attentions))
        for resnet, attn in blocks:
            hidden_states = resnet(
                hidden_states,
                temb,
                image_only_indicator=image_only_indicator,
            )
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                image_only_indicator=image_only_indicator,
                return_dict=False,
            )[0]

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class UpBlockSpatioTemporal(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        resolution_idx: Optional[int] = None,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        add_upsample: bool = True,
        dtype: str = "float16",
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                SpatioTemporalResBlock(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    dtype=dtype,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [
                    Upsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        dtype=dtype,
                    )
                ]
            )
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx

    def forward(
        self,
        hidden_states: Tensor,
        res_hidden_states_tuple: Tuple[Tensor, ...],
        temb: Optional[Tensor] = None,
        image_only_indicator: Optional[Tensor] = None,
    ) -> Tensor:
        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            hidden_states = ops.concatenate()(
                [hidden_states, res_hidden_states], dim=-1
            )

            hidden_states = resnet(
                hidden_states,
                temb,
                image_only_indicator=image_only_indicator,
            )

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class CrossAttnUpBlockSpatioTemporal(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        resolution_idx: Optional[int] = None,
        num_layers: int = 1,
        transformer_layers_per_block: Union[int, Tuple[int]] = 1,
        resnet_eps: float = 1e-6,
        num_attention_heads: int = 1,
        cross_attention_dim: int = 1280,
        add_upsample: bool = True,
        dtype: str = "float16",
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                SpatioTemporalResBlock(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    dtype=dtype,
                )
            )
            attentions.append(
                TransformerSpatioTemporalModel(
                    num_attention_heads,
                    out_channels // num_attention_heads,
                    in_channels=out_channels,
                    num_layers=transformer_layers_per_block[i],
                    cross_attention_dim=cross_attention_dim,
                    dtype=dtype,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [
                    Upsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        dtype=dtype,
                    )
                ]
            )
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx

    def forward(
        self,
        hidden_states: Tensor,
        res_hidden_states_tuple: Tuple[Tensor, ...],
        temb: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        image_only_indicator: Optional[Tensor] = None,
    ) -> Tensor:
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            hidden_states = ops.concatenate()(
                [hidden_states, res_hidden_states], dim=-1
            )

            hidden_states = resnet(
                hidden_states,
                temb,
                image_only_indicator=image_only_indicator,
            )
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                image_only_indicator=image_only_indicator,
                return_dict=False,
            )[0]

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states
