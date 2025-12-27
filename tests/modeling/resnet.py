import unittest

from typing import cast, List, Optional, Tuple, Union

import diffusers.models.resnet as resnet_torch

import torch
from dinoml.compiler import compile_model, ops
from dinoml.frontend import Tensor
from dinoml.testing import detect_target
from dinoml.testing.test_utils import get_random_torch_tensor

import dinoml.modeling.diffusers.resnet as resnet
from dinoml.builder.config import mark_output


class ResnetTestCase(unittest.TestCase):
    def _test_resnet_block_cond_norm_2d(
        self,
        shape: List[int],
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        groups_out: Optional[int] = None,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        time_embedding_norm: str = "ada_group",
        output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None,
        up: bool = False,
        down: bool = False,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: Optional[int] = None,
        dtype: str = "float16",
        tolerance: float = 1e-5,
    ):
        batch, channels, height, width = shape

        x = get_random_torch_tensor(shape, dtype=dtype)
        temb = get_random_torch_tensor(
            (
                [1, temb_channels, 1, 1]
                if time_embedding_norm == "spatial"
                else [1, temb_channels]
            ),
            dtype=dtype,
        )
        x_dinoml = x.clone().permute(0, 2, 3, 1).contiguous().to(x.device, x.dtype)
        temb_dinoml = temb.clone().to(temb.device, temb.dtype)
        if time_embedding_norm == "spatial":
            temb_dinoml = temb_dinoml.permute(0, 2, 3, 1).contiguous()

        op = (
            resnet_torch.ResnetBlockCondNorm2D(
                in_channels=in_channels,
                out_channels=out_channels,
                conv_shortcut=conv_shortcut,
                dropout=dropout,
                temb_channels=temb_channels,
                groups=in_channels // groups,
                groups_out=groups_out,
                eps=eps,
                non_linearity=non_linearity,
                time_embedding_norm=time_embedding_norm,
                output_scale_factor=output_scale_factor,
                use_in_shortcut=use_in_shortcut,
                up=up,
                down=down,
                conv_shortcut_bias=conv_shortcut_bias,
                conv_2d_out_channels=conv_2d_out_channels,
            )
            .eval()
            .to(x.device, x.dtype)
        )

        state_dict_pt = cast(dict[str, torch.Tensor], op.state_dict())
        state_dict_dinoml = {}
        for key, value in state_dict_pt.items():
            key_dinoml = key.replace(".", "_")
            if value.ndim == 4 and "weight" in key:
                value = value.permute(0, 2, 3, 1).contiguous()
            value = value.to(x.device, x.dtype)
            state_dict_dinoml[key_dinoml] = value

        with torch.inference_mode():
            y_pt = op.forward(x, temb)

        y = torch.empty_like(y_pt.permute(0, 2, 3, 1).contiguous()).to(
            x.device, x.dtype
        )

        X = Tensor(
            shape=[batch, height, width, channels],
            dtype=dtype,
            name="X",
            is_input=True,
        )
        Temb = Tensor(
            shape=(
                [1, 1, 1, temb_channels]
                if time_embedding_norm == "spatial"
                else [1, temb_channels]
            ),
            dtype=dtype,
            name="Temb",
            is_input=True,
        )

        op_dinoml = resnet.ResnetBlockCondNorm2D(
            in_channels=in_channels,
            out_channels=out_channels,
            conv_shortcut=conv_shortcut,
            dropout=dropout,
            temb_channels=temb_channels,
            groups=in_channels // groups,
            groups_out=groups_out,
            eps=eps,
            non_linearity=non_linearity,
            time_embedding_norm=time_embedding_norm,
            output_scale_factor=output_scale_factor,
            use_in_shortcut=use_in_shortcut,
            up=up,
            down=down,
            conv_shortcut_bias=conv_shortcut_bias,
            conv_2d_out_channels=conv_2d_out_channels,
            dtype=dtype,
        )
        op_dinoml.name_parameter_tensor()
        Y = op_dinoml.forward(X, Temb)
        Y = mark_output(Y, "Y")

        target = detect_target()
        test_name = f"test_resnet_block_cond_norm_2d_{dtype}_in_channels{in_channels}"
        if time_embedding_norm is not None:
            test_name += f"_{time_embedding_norm}"
        if conv_shortcut:
            test_name += "_conv_shortcut"
        if up:
            test_name += "_up"
        if down:
            test_name += "_down"

        x = {"X": x_dinoml, "Temb": temb_dinoml}
        module = compile_model(
            Y,
            target,
            "./tmp",
            test_name,
            constants=state_dict_dinoml,
        )
        module.run_with_tensors(x, [y])
        y = y.permute(0, 3, 1, 2).contiguous()
        torch.testing.assert_close(
            y,
            y_pt.to(y.dtype),
            rtol=tolerance,
            atol=tolerance,
            msg=lambda msg: f"{msg}\n\npt ({y_pt.shape}):\n{y_pt}\n\ndinoml ({y.shape}):\n{y}\n\n",
        )

    def _test_resnet_block_2d(
        self,
        shape: List[int],
        temb_shape: Optional[List[int]],
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        groups_out: Optional[int] = None,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        skip_time_act: bool = False,
        time_embedding_norm: str = "default",  # default, scale_shift
        output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None,
        up: bool = False,
        down: bool = False,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: Optional[int] = None,
        dtype: str = "float16",
        tolerance: float = 1e-5,
    ):
        batch, channels, height, width = shape
        x = get_random_torch_tensor(shape, dtype=dtype)
        temb = (
            get_random_torch_tensor(temb_shape, dtype=dtype)
            if temb_shape is not None
            else None
        )
        x_dinoml = x.clone().permute(0, 2, 3, 1).contiguous().to(x.device, x.dtype)
        temb_dinoml = (
            temb.clone().to(temb.device, temb.dtype) if temb is not None else None
        )

        op = (
            resnet_torch.ResnetBlock2D(
                in_channels=in_channels,
                out_channels=out_channels,
                conv_shortcut=conv_shortcut,
                dropout=dropout,
                temb_channels=temb_channels if temb is not None else None,
                groups=groups,
                groups_out=groups_out,
                eps=eps,
                non_linearity=non_linearity,
                skip_time_act=skip_time_act,
                time_embedding_norm=time_embedding_norm,
                output_scale_factor=output_scale_factor,
                use_in_shortcut=use_in_shortcut,
                up=up,
                down=down,
                conv_shortcut_bias=conv_shortcut_bias,
                conv_2d_out_channels=conv_2d_out_channels,
            )
            .eval()
            .to(x.device, x.dtype)
        )

        state_dict_pt = cast(dict[str, torch.Tensor], op.state_dict())
        state_dict_dinoml = {}
        for key, value in state_dict_pt.items():
            key_dinoml = key.replace(".", "_")
            if value.ndim == 4 and "weight" in key:
                value = value.permute(0, 2, 3, 1).contiguous()
            value = value.to(x.device, x.dtype)
            state_dict_dinoml[key_dinoml] = value

        with torch.inference_mode():
            y_pt = op.forward(x, temb)

        y = torch.empty_like(y_pt.permute(0, 2, 3, 1).contiguous()).to(
            x.device, x.dtype
        )

        X = Tensor(
            shape=[batch, height, width, channels],
            dtype=dtype,
            name="X",
            is_input=True,
        )
        if temb_shape is not None:
            Temb = Tensor(
                shape=temb_shape,
                dtype=dtype,
                name="Temb",
                is_input=True,
            )
        else:
            Temb = None

        op_dinoml = resnet.ResnetBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            conv_shortcut=conv_shortcut,
            dropout=dropout,
            temb_channels=temb_channels if temb is not None else None,
            groups=groups,
            groups_out=groups_out,
            eps=eps,
            non_linearity=non_linearity,
            skip_time_act=skip_time_act,
            time_embedding_norm=time_embedding_norm,
            output_scale_factor=output_scale_factor,
            use_in_shortcut=use_in_shortcut,
            up=up,
            down=down,
            conv_shortcut_bias=conv_shortcut_bias,
            conv_2d_out_channels=conv_2d_out_channels,
            dtype=dtype,
        )
        op_dinoml.name_parameter_tensor()
        Y = op_dinoml.forward(X, Temb)
        Y = mark_output(Y, "Y")

        target = detect_target()
        test_name = f"test_resnet_block_2d_{dtype}_in_channels{in_channels}"
        if time_embedding_norm is not None:
            test_name += f"_{time_embedding_norm}"
        if conv_shortcut:
            test_name += "_conv_shortcut"
        if up:
            test_name += "_up"
        if down:
            test_name += "_down"
        x = {"X": x_dinoml}
        if Temb is not None:
            x["Temb"] = temb_dinoml
        module = compile_model(
            Y,
            target,
            "./tmp",
            test_name,
            constants=state_dict_dinoml,
        )
        module.run_with_tensors(x, [y])
        y = y.permute(0, 3, 1, 2).contiguous()
        torch.testing.assert_close(
            y,
            y_pt.to(y.dtype),
            rtol=tolerance,
            atol=tolerance,
            msg=lambda msg: f"{msg}\n\npt ({y_pt.shape}):\n{y_pt}\n\ndinoml ({y.shape}):\n{y}\n\n",
        )

    def _test_conv1d_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        n_groups: int = 8,
        activation: str = "mish",
        dtype: str = "float16",
        shape: Tuple[int, int, int] = (1, 64, 32),
        tolerance: float = 1e-5,
    ):
        batch, channels, seq_len = shape

        x = get_random_torch_tensor(shape, dtype=dtype)
        x_dinoml = x.clone().permute(0, 2, 1).contiguous().to(x.device, x.dtype)

        op = (
            resnet_torch.Conv1dBlock(
                inp_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                n_groups=n_groups,
                activation=activation,
            )
            .eval()
            .to(x.device, x.dtype)
        )

        state_dict_pt = cast(dict[str, torch.Tensor], op.state_dict())
        state_dict_dinoml = {}
        for key, value in state_dict_pt.items():
            key_dinoml = key.replace(".", "_")
            if "conv" in key.lower() and "weight" in key:
                value = value.permute(0, 2, 1).contiguous()
            value = value.to(x.device, x.dtype)
            state_dict_dinoml[key_dinoml] = value

        with torch.inference_mode():
            y_pt = op.forward(x)

        y = torch.empty_like(y_pt.permute(0, 2, 1).contiguous())

        X = Tensor(
            shape=[batch, seq_len, channels],
            dtype=dtype,
            name="X",
            is_input=True,
        )

        op_dinoml = resnet.Conv1dBlock(
            inp_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            n_groups=n_groups,
            activation=activation,
            dtype=dtype,
        )
        op_dinoml.name_parameter_tensor()
        Y = op_dinoml.forward(X)
        Y = mark_output(Y, "Y")

        target = detect_target()
        test_name = f"test_conv1d_block_{dtype}_in_channels{in_channels}_out_channels{out_channels}"
        x = {"X": x_dinoml}
        module = compile_model(
            Y,
            target,
            "./tmp",
            test_name,
            constants=state_dict_dinoml,
        )
        module.run_with_tensors(x, [y])
        y = y.permute(0, 2, 1).contiguous()
        torch.testing.assert_close(
            y,
            y_pt.to(y.dtype),
            rtol=tolerance,
            atol=tolerance,
            msg=lambda msg: f"{msg}\n\npt ({y_pt.shape}):\n{y_pt}\n\ndinoml ({y.shape}):\n{y}\n\n",
        )

    def _test_residual_temporal_block_1d(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        activation: str = "mish",
        dtype: str = "float16",
        shape: Tuple[int, int, int] = (1, 64, 32),
        tolerance: float = 1e-5,
    ):
        batch, channels, seq_len = shape

        x = get_random_torch_tensor(shape, dtype=dtype)
        x_dinoml = x.clone().permute(0, 2, 1).contiguous().to(x.device, x.dtype)
        temb = get_random_torch_tensor([1, embed_dim], dtype=dtype)
        temb_dinoml = temb.clone()

        op = (
            resnet_torch.ResidualTemporalBlock1D(
                inp_channels=in_channels,
                out_channels=out_channels,
                embed_dim=embed_dim,
                kernel_size=kernel_size,
                activation=activation,
            )
            .eval()
            .to(x.device, x.dtype)
        )

        state_dict_pt = cast(dict[str, torch.Tensor], op.state_dict())
        state_dict_dinoml = {}
        for key, value in state_dict_pt.items():
            key_dinoml = key.replace(".", "_")
            if "conv" in key.lower() and "weight" in key and value.ndim == 3:
                value = value.permute(0, 2, 1).contiguous()
            value = value.to(x.device, x.dtype)
            state_dict_dinoml[key_dinoml] = value

        with torch.inference_mode():
            y_pt = op.forward(x, temb)

        y = torch.empty_like(y_pt.permute(0, 2, 1).contiguous())

        X = Tensor(
            shape=[batch, seq_len, channels],
            dtype=dtype,
            name="X",
            is_input=True,
        )
        Temb = Tensor(
            shape=[1, embed_dim],
            dtype=dtype,
            name="Temb",
            is_input=True,
        )

        op_dinoml = resnet.ResidualTemporalBlock1D(
            inp_channels=in_channels,
            out_channels=out_channels,
            embed_dim=embed_dim,
            kernel_size=kernel_size,
            activation=activation,
            dtype=dtype,
        )
        op_dinoml.name_parameter_tensor()
        Y = op_dinoml.forward(X, Temb)
        Y = mark_output(Y, "Y")

        target = detect_target()
        test_name = f"test_residual_temporal_block_1d_{dtype}_in_channels{in_channels}_out_channels{out_channels}_dim{embed_dim}"
        x = {"X": x_dinoml, "Temb": temb_dinoml}
        module = compile_model(
            Y,
            target,
            "./tmp",
            test_name,
            constants=state_dict_dinoml,
        )
        module.run_with_tensors(x, [y])
        y = y.permute(0, 2, 1).contiguous()
        torch.testing.assert_close(
            y,
            y_pt.to(y.dtype),
            rtol=tolerance,
            atol=tolerance,
            msg=lambda msg: f"{msg}\n\npt ({y_pt.shape}):\n{y_pt}\n\ndinoml ({y.shape}):\n{y}\n\n",
        )

    def _test_temporal_conv_layer(
        self,
        shape: List[int],
        in_dim: int,
        out_dim: Optional[int] = None,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        dtype: str = "float16",
        num_frames: int = 1,
        tolerance: float = 1e-5,
    ):
        batch, channels, height, width = shape
        x = get_random_torch_tensor(shape, dtype=dtype)
        x_dinoml = x.clone().permute(0, 2, 3, 1).contiguous().to(x.device, x.dtype)

        op = (
            resnet_torch.TemporalConvLayer(
                in_dim=in_dim,
                out_dim=out_dim,
                dropout=dropout,
                norm_num_groups=norm_num_groups,
            )
            .eval()
            .to(x.device, x.dtype)
        )

        state_dict_pt = cast(dict[str, torch.Tensor], op.state_dict())
        state_dict_dinoml = {}
        for key, value in state_dict_pt.items():
            key_dinoml = key.replace(".", "_")
            if "conv" in key.lower() and "weight" in key and value.ndim == 4:
                value = value.permute(0, 2, 3, 1).contiguous()
            value = value.to(x.device, x.dtype)
            state_dict_dinoml[key_dinoml] = value

        with torch.inference_mode():
            y_pt = op.forward(x, num_frames)

        y = torch.empty_like(y_pt.permute(0, 2, 3, 1).contiguous()).to(
            x.device, x.dtype
        )

        X = Tensor(
            shape=[batch, height, width, channels],
            dtype=dtype,
            name="X",
            is_input=True,
        )

        op_dinoml = resnet.TemporalConvLayer(
            in_dim=in_dim,
            out_dim=out_dim,
            dropout=dropout,
            norm_num_groups=norm_num_groups,
            dtype=dtype,
        )
        op_dinoml.name_parameter_tensor()
        Y = op_dinoml.forward(X, num_frames)
        Y = mark_output(Y, "Y")

        target = detect_target()
        test_name = (
            f"test_temporal_conv_layer_{dtype}_in_dim{in_dim}_frames{num_frames}"
        )
        inputs_dict = {"X": x_dinoml}
        module = compile_model(
            Y,
            target,
            "./tmp",
            test_name,
            constants=state_dict_dinoml,
        )
        module.run_with_tensors(inputs_dict, [y])
        y = y.permute(0, 3, 1, 2).contiguous()
        torch.testing.assert_close(
            y,
            y_pt.to(y.dtype),
            rtol=tolerance,
            atol=tolerance,
            msg=lambda msg: f"{msg}\n\npt ({y_pt.shape}):\n{y_pt}\n\ndinoml ({y.shape}):\n{y}\n\n",
        )

    def _test_temporal_resnet_block(
        self,
        shape: List[int],
        temb_shape: List[int],
        in_channels: int,
        out_channels: Optional[int] = None,
        temb_channels: int = 512,
        eps: float = 1e-6,
        dtype: str = "float16",
        tolerance: float = 1e-5,
    ):
        batch, channels, frames, height, width = shape
        x = get_random_torch_tensor(shape, dtype=dtype)
        temb = get_random_torch_tensor(temb_shape, dtype=dtype)
        x_dinoml = x.clone().permute(0, 2, 3, 4, 1).contiguous().to(x.device, x.dtype)
        temb_dinoml = temb.clone().to(temb.device, temb.dtype)

        op = (
            resnet_torch.TemporalResnetBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                eps=eps,
            )
            .eval()
            .to(x.device, x.dtype)
        )

        state_dict_pt = cast(dict[str, torch.Tensor], op.state_dict())
        state_dict_dinoml = {}
        for key, value in state_dict_pt.items():
            key_dinoml = key.replace(".", "_")
            if "weight" in key and value.ndim == 5:
                value = value.permute(0, 2, 3, 4, 1).contiguous()
            value = value.to(x.device, x.dtype)
            state_dict_dinoml[key_dinoml] = value

        with torch.inference_mode():
            y_pt = op.forward(x, temb)

        y = torch.empty_like(y_pt.permute(0, 2, 3, 4, 1).contiguous()).to(
            x.device, x.dtype
        )

        X = Tensor(
            shape=[batch, frames, height, width, channels],
            dtype=dtype,
            name="X",
            is_input=True,
        )
        Temb = Tensor(
            shape=temb_shape,
            dtype=dtype,
            name="Temb",
            is_input=True,
        )

        op_dinoml = resnet.TemporalResnetBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            eps=eps,
            dtype=dtype,
        )
        op_dinoml.name_parameter_tensor()
        Y = op_dinoml.forward(X, Temb)
        Y = mark_output(Y, "Y")

        target = detect_target()
        test_name = (
            f"test_temporal_resnet_block_{dtype}_in_dim{in_channels}_frames{frames}"
        )
        inputs_dict = {"X": x_dinoml, "Temb": temb_dinoml}
        module = compile_model(
            Y,
            target,
            "./tmp",
            test_name,
            constants=state_dict_dinoml,
        )
        module.run_with_tensors(inputs_dict, [y])
        y = y.permute(0, 4, 1, 2, 3).contiguous()
        torch.testing.assert_close(
            y,
            y_pt.to(y.dtype),
            rtol=tolerance,
            atol=tolerance,
            msg=lambda msg: f"{msg}\n\npt ({y_pt.shape}):\n{y_pt}\n\ndinoml ({y.shape}):\n{y}\n\n",
        )

    def _test_alpha_blender(
        self,
        shape: List[int],
        alpha: float,
        merge_strategy: str = "learned_with_images",
        switch_spatial_to_temporal_mix: bool = False,
        dtype: str = "float16",
        tolerance: float = 1e-5,
    ):
        if len(shape) == 5:
            batch, channels, frames, height, width = shape
        x_spatial = get_random_torch_tensor(shape, dtype=dtype)
        x_temporal = get_random_torch_tensor(shape, dtype=dtype)
        image_only_indicator = (
            torch.randint(
                0,
                1,
                [1, shape[0]] if len(shape) != 5 else [shape[0], frames],
                dtype=torch.bool,
                device=x_spatial.device,
            )
            if merge_strategy == "learned_with_images"
            else None
        )
        x_spatial_dinoml = x_spatial.clone().to(x_spatial.device, x_spatial.dtype)
        x_temporal_dinoml = x_temporal.clone().to(x_temporal.device, x_temporal.dtype)
        if len(shape) == 5:
            x_spatial_dinoml = x_spatial_dinoml.permute(0, 2, 3, 4, 1).contiguous()
            x_temporal_dinoml = x_temporal_dinoml.permute(0, 2, 3, 4, 1).contiguous()
        image_only_indicator_dinoml = (
            image_only_indicator.clone().to(
                image_only_indicator.device, image_only_indicator.dtype
            )
            if image_only_indicator is not None
            else None
        )

        op = (
            resnet_torch.AlphaBlender(
                alpha=alpha,
                merge_strategy=merge_strategy,
                switch_spatial_to_temporal_mix=switch_spatial_to_temporal_mix,
            )
            .eval()
            .to(x_spatial.device, x_spatial.dtype)
        )

        state_dict_pt = cast(dict[str, torch.Tensor], op.state_dict())
        state_dict_dinoml = {}
        for key, value in state_dict_pt.items():
            key_dinoml = key.replace(".", "_")
            if "weight" in key and value.ndim == 5:
                value = value.permute(0, 2, 3, 4, 1).contiguous()
            value = value.to(x_spatial.device, x_spatial.dtype)
            state_dict_dinoml[key_dinoml] = value

        if merge_strategy == "fixed":
            state_dict_dinoml["mix_factor"] = torch.tensor(
                [alpha], dtype=x_spatial.dtype, device=x_spatial.device
            )

        with torch.inference_mode():
            y_pt = op.forward(x_spatial, x_temporal, image_only_indicator)

        y = torch.empty_like(y_pt).to(x_spatial.device, x_spatial.dtype)
        if len(shape) == 5:
            y = y.permute(0, 2, 3, 4, 1).contiguous()

        if len(shape) == 5:
            shape = [batch, frames, height, width, channels]
        X_spatial = Tensor(
            shape=shape,
            dtype=dtype,
            name="X_spatial",
            is_input=True,
        )
        X_temporal = Tensor(
            shape=shape,
            dtype=dtype,
            name="X_temporal",
            is_input=True,
        )
        Image_only_indicator = (
            Tensor(
                shape=[1, shape[0]] if len(shape) != 5 else [shape[0], frames],
                dtype="bool",
                name="Image_only_indicator",
                is_input=True,
            )
            if image_only_indicator is not None
            else None
        )

        op_dinoml = resnet.AlphaBlender(
            alpha=alpha,
            merge_strategy=merge_strategy,
            switch_spatial_to_temporal_mix=switch_spatial_to_temporal_mix,
            dtype=dtype,
        )
        op_dinoml.name_parameter_tensor()
        Y = op_dinoml.forward(X_spatial, X_temporal, Image_only_indicator)
        Y = mark_output(Y, "Y")

        target = detect_target()
        test_name = f"test_alpha_blender_{dtype}_merge_strategy_{merge_strategy}"
        inputs_dict = {"X_spatial": x_spatial_dinoml, "X_temporal": x_temporal_dinoml}
        if Image_only_indicator is not None:
            inputs_dict["Image_only_indicator"] = image_only_indicator_dinoml
        module = compile_model(
            Y,
            target,
            "./tmp",
            test_name,
            constants=state_dict_dinoml,
        )
        module.run_with_tensors(inputs_dict, [y])
        if len(shape) == 5:
            y = y.permute(0, 4, 1, 2, 3).contiguous()
        torch.testing.assert_close(
            y,
            y_pt.to(y.dtype),
            rtol=tolerance,
            atol=tolerance,
            msg=lambda msg: f"{msg}\n\npt ({y_pt.shape}):\n{y_pt}\n\ndinoml ({y.shape}):\n{y}\n\n",
        )

    def _test_spatiotemporal_res_block(
        self,
        shape: List[int],
        num_frames: int,
        in_channels: int,
        out_channels: Optional[int] = None,
        temb_channels: int = 512,
        eps: float = 1e-6,
        temporal_eps: Optional[float] = None,
        merge_factor: float = 0.5,
        merge_strategy: str = "learned_with_images",
        switch_spatial_to_temporal_mix: bool = False,
        dtype: str = "float16",
        tolerance: float = 1e-5,
    ):
        batch_frames, channels, height, width = shape
        batch = batch_frames // num_frames
        temb_shape = [batch, num_frames, temb_channels]
        x = get_random_torch_tensor(shape, dtype=dtype)
        # temb = get_random_torch_tensor(temb_shape, dtype=dtype)
        image_only_indicator = torch.randint(
            0,
            1,
            [batch, num_frames],
            dtype=torch.bool,
            device=x.device,
        )
        x_dinoml = x.clone().permute(0, 2, 3, 1).contiguous().to(x.device, x.dtype)
        # temb_dinoml = temb.clone().to(temb.device, temb.dtype)
        image_only_indicator_dinoml = image_only_indicator.clone().to(
            image_only_indicator.device, image_only_indicator.dtype
        )

        op = (
            resnet_torch.SpatioTemporalResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                eps=eps,
                temporal_eps=temporal_eps,
                merge_factor=merge_factor,
                merge_strategy=merge_strategy,
                switch_spatial_to_temporal_mix=switch_spatial_to_temporal_mix,
            )
            .eval()
            .to(x.device, x.dtype)
        )

        state_dict_pt = cast(dict[str, torch.Tensor], op.state_dict())
        state_dict_dinoml = {}
        for key, value in state_dict_pt.items():
            key_dinoml = key.replace(".", "_")
            if "weight" in key and value.ndim == 4:
                value = value.permute(0, 2, 3, 1).contiguous()
            if "weight" in key and value.ndim == 5:
                value = value.permute(0, 2, 3, 4, 1).contiguous()
            value = value.to(x.device, x.dtype)
            state_dict_dinoml[key_dinoml] = value

        if merge_strategy == "fixed":
            state_dict_dinoml["mix_factor"] = torch.tensor(
                [merge_factor], dtype=x.dtype, device=x.device
            )

        with torch.inference_mode():
            y_pt = op.forward(x, None, image_only_indicator)
        print(y_pt.shape)

        y = torch.empty_like(y_pt.permute(0, 2, 3, 1).contiguous()).to(
            x.device, x.dtype
        )

        X = Tensor(
            shape=[batch_frames, height, width, channels],
            dtype=dtype,
            name="X",
            is_input=True,
        )
        # Temb = Tensor(
        #     shape=temb_shape,
        #     dtype=dtype,
        #     name="Temb",
        #     is_input=True,
        # )
        Image_only_indicator = Tensor(
            shape=[batch, num_frames],
            dtype="bool",
            name="Image_only_indicator",
            is_input=True,
        )

        op_dinoml = resnet.SpatioTemporalResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            eps=eps,
            temporal_eps=temporal_eps,
            merge_factor=merge_factor,
            merge_strategy=merge_strategy,
            switch_spatial_to_temporal_mix=switch_spatial_to_temporal_mix,
            dtype=dtype,
        )
        op_dinoml.name_parameter_tensor()
        Y = op_dinoml.forward(X, None, Image_only_indicator)
        Y = mark_output(Y, "Y")

        target = detect_target()
        test_name = (
            f"test_spatiotemporal_res_block_{dtype}_merge_strategy_{merge_strategy}"
        )
        inputs_dict = {"X": x_dinoml}
        module = compile_model(
            Y,
            target,
            "./tmp",
            test_name,
            constants=state_dict_dinoml,
        )
        module.run_with_tensors(inputs_dict, [y])
        if y.ndim == 4:
            y = y.permute(0, 3, 1, 2).contiguous()
        torch.testing.assert_close(
            y,
            y_pt.to(y.dtype),
            rtol=tolerance,
            atol=tolerance,
            msg=lambda msg: f"{msg}\n\npt ({y_pt.shape}):\n{y_pt}\n\ndinoml ({y.shape}):\n{y}\n\n",
        )

    def test_resnet_block_cond_norm_2d(self):
        self._test_resnet_block_cond_norm_2d(
            shape=[1, 1280, 64, 64],
            in_channels=1280,
            temb_channels=1280,
            time_embedding_norm="spatial",
            tolerance=3e-3,
            dtype="float16",
        )
        self._test_resnet_block_cond_norm_2d(
            shape=[1, 1280, 64, 64],
            in_channels=1280,
            temb_channels=1280,
            tolerance=3e-3,
            up=True,
            dtype="float16",
        )
        # FIXME: Input tensor elementwise_15_0 not established in graph for op avg_pool2d_1
        # `hidden_states = x`
        # `hidden_states = self.nonlinearity(hidden_states)` # elementwise_15_0
        # `x = self.downsample(x)` # seems to override `hidden_states`
        # `hidden_states = self.downsample(hidden_states)` # not established
        # self._test_resnet_block_cond_norm_2d(
        #     shape=[1, 1280, 64, 64],
        #     in_channels=1280,
        #     temb_channels=1280,
        #     tolerance=3e-3,
        #     down=True,
        #     dtype="float16",
        # )

    def test_resnet_block_2d(self):
        self._test_resnet_block_2d(
            shape=[1, 64, 32, 32],
            temb_shape=[1, 512],
            in_channels=64,
            conv_shortcut=True,
            temb_channels=512,
            groups=32,
            groups_out=None,
            eps=1e-6,
            non_linearity="swish",
            skip_time_act=False,
            time_embedding_norm="default",
            output_scale_factor=1.0,
            use_in_shortcut=True,
            up=False,
            down=False,
            conv_shortcut_bias=True,
            conv_2d_out_channels=None,
            tolerance=2e-3,
            dtype="float16",
        )
        self._test_resnet_block_2d(
            shape=[1, 128, 32, 32],
            temb_shape=[1, 512],
            in_channels=128,
            conv_shortcut=False,
            temb_channels=512,
            groups=32,
            groups_out=None,
            eps=1e-6,
            non_linearity="swish",
            skip_time_act=False,
            time_embedding_norm="scale_shift",
            output_scale_factor=1.0,
            use_in_shortcut=True,
            up=True,
            down=False,
            conv_shortcut_bias=True,
            conv_2d_out_channels=None,
            tolerance=3e-3,
            dtype="float16",
        )
        # FIXME: Input tensor elementwise_3_0 not established in graph for op avg_pool2d_0
        # self._test_resnet_block_2d(
        #     shape=[1, 64, 64, 64],
        #     temb_shape=[1, 512],
        #     in_channels=64,
        #     conv_shortcut=True,
        #     temb_channels=512,
        #     groups=32,
        #     groups_out=None,
        #     eps=1e-6,
        #     non_linearity="swish",
        #     skip_time_act=False,
        #     time_embedding_norm="default",
        #     output_scale_factor=1.0,
        #     use_in_shortcut=False,
        #     up=False,
        #     down=True,
        #     conv_shortcut_bias=True,
        #     conv_2d_out_channels=None,
        #     tolerance=3e-3,
        #     dtype="float16",
        # )
        self._test_resnet_block_2d(
            shape=[1, 64, 64, 64],
            temb_shape=None,
            in_channels=64,
            conv_shortcut=True,
            temb_channels=None,
            groups=32,
            groups_out=None,
            eps=1e-6,
            non_linearity="swish",
            skip_time_act=False,
            time_embedding_norm="default",
            output_scale_factor=1.0,
            use_in_shortcut=False,
            up=False,
            down=False,
            conv_shortcut_bias=True,
            conv_2d_out_channels=None,
            tolerance=3e-3,
            dtype="float16",
        )

    def test_conv1d_block(self):
        self._test_conv1d_block(
            in_channels=14,
            out_channels=8,
            kernel_size=5,
            dtype="float16",
            shape=(1, 14, 32),
            tolerance=3e-3,
        )

    def test_residual_temporal_block_1d(self):
        self._test_residual_temporal_block_1d(
            in_channels=14,
            out_channels=8,
            embed_dim=512,
            kernel_size=5,
            dtype="float16",
            shape=(1, 14, 32),
            tolerance=3e-3,
        )

    def test_temporal_conv_layer(self):
        self._test_temporal_conv_layer(
            shape=[1, 320, 64, 64],
            in_dim=320,
            dtype="float16",
            num_frames=1,
            tolerance=1e-3,
        )
        self._test_temporal_conv_layer(
            shape=[2, 320, 64, 64],
            in_dim=320,
            dtype="float16",
            num_frames=2,
            tolerance=1e-3,
        )

    def test_temporal_resnet_block(self):
        self._test_temporal_resnet_block(
            shape=[1, 32, 4, 48, 48],
            temb_shape=[1, 4, 512],
            in_channels=32,
            temb_channels=512,
            dtype="float16",
            tolerance=3e-3,
        )

    def test_alpha_blender_fixed(self):
        shapes = [[1, 64, 16, 32, 32], [16, 1024, 64]]
        merge_strategies = ["fixed", "learned", "learned_with_images"]
        for shape in shapes:
            for merge_strategy in merge_strategies:
                self._test_alpha_blender(
                    shape=shape,
                    alpha=0.5,
                    merge_strategy=merge_strategy,
                    dtype="float16",
                    tolerance=1e-3,
                )

    def test_spatiotemporal_res_block(self):
        # TODO: more cases
        self._test_spatiotemporal_res_block(
            shape=[16, 64, 32, 32],
            num_frames=16,
            in_channels=64,
            temb_channels=None,
            merge_strategy="learned",
            switch_spatial_to_temporal_mix=True,
            dtype="float16",
            tolerance=3e-3,
        )


if __name__ == "__main__":
    unittest.main()
