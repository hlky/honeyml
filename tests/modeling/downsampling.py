import unittest

from typing import cast, List, Literal, Optional, Tuple

import diffusers.models.downsampling as downsampling_torch

import torch
from dinoml.compiler import compile_model, ops
from dinoml.frontend import Tensor
from dinoml.testing import detect_target
from dinoml.testing.test_utils import get_random_torch_tensor

import dinoml.modeling.diffusers.downsampling as downsampling
from dinoml.builder.config import mark_output


# TODO: FirDownsample2D
# TODO: KDownsample2D


class DownsamplingTestCase(unittest.TestCase):
    def _test_downsample1d(
        self,
        shape: Tuple[int, int, int],
        use_conv: bool = False,
        dtype: str = "float16",
        tolerance: float = 1e-5,
    ):
        batch, channels, seq_len = shape

        x = get_random_torch_tensor(shape, dtype=dtype)
        op = (
            downsampling_torch.Downsample1D(
                channels=channels,
                use_conv=use_conv,
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
        print(y_pt.shape)
        x = x.permute(0, 2, 1).contiguous()
        y = torch.empty_like(y_pt.permute(0, 2, 1).contiguous())

        X = Tensor(
            shape=[batch, seq_len, channels], dtype=dtype, name="X", is_input=True
        )
        op = downsampling.Downsample1D(channels=channels, use_conv=use_conv)
        op.name_parameter_tensor()
        Y = op.forward(X)
        Y = mark_output(Y, "Y")
        target = detect_target()
        module = compile_model(
            Y,
            target,
            "./tmp",
            f"test_downsample1d_c-{channels}_conv-{use_conv}",
            constants=state_dict_dinoml,
        )

        module.run_with_tensors([x], [y])
        y = y.permute(0, 2, 1).contiguous()
        torch.testing.assert_close(
            y,
            y_pt,
            rtol=tolerance,
            atol=tolerance,
            msg=lambda msg: f"{msg}\n\npt ({y_pt.shape}):\n{y_pt}\n\ndinoml ({y.shape}):\n{y}\n\n",
        )

    def _test_downsample2d(
        self,
        shape: Tuple[int, int, int, int],
        use_conv: bool = False,
        norm_type: Optional[Literal["ln_norm", "rms_norm"]] = None,
        eps: float = 1e-5,
        dtype: str = "float16",
        tolerance: float = 1e-5,
    ):
        batch, channels, height, width = shape

        x = get_random_torch_tensor(shape, dtype=dtype)
        op = (
            downsampling_torch.Downsample2D(
                channels=channels,
                use_conv=use_conv,
                norm_type=norm_type,
                eps=eps,
            )
            .eval()
            .to(x.device, x.dtype)
        )
        state_dict_pt = cast(dict[str, torch.Tensor], op.state_dict())
        state_dict_dinoml = {}
        for key, value in state_dict_pt.items():
            key_dinoml = key.replace(".", "_")
            if "conv" in key.lower() and "weight" in key:
                value = value.permute(0, 2, 3, 1).contiguous()
            value = value.to(x.device, x.dtype)
            state_dict_dinoml[key_dinoml] = value

        y_pt = op.forward(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        y = torch.empty_like(y_pt.permute(0, 2, 3, 1).contiguous())

        X = Tensor(
            shape=[batch, height, width, channels], dtype=dtype, name="X", is_input=True
        )
        op = downsampling.Downsample2D(
            channels=channels,
            use_conv=use_conv,
            norm_type=norm_type,
            eps=eps,
        )
        op.name_parameter_tensor()
        Y = op.forward(X)
        Y = mark_output(Y, "Y")
        target = detect_target()
        module = compile_model(
            Y,
            target,
            "./tmp",
            f"test_downsample2d_c-{channels}_conv-{use_conv}_norm-{norm_type}",
            constants=state_dict_dinoml,
        )

        module.run_with_tensors([x], [y])
        y = y.permute(0, 3, 1, 2).contiguous()
        torch.testing.assert_close(
            y,
            y_pt,
            rtol=tolerance,
            atol=tolerance,
            msg=lambda msg: f"{msg}\n\npt ({y_pt.shape}):\n{y_pt}\n\ndinoml ({y.shape}):\n{y}\n\n",
        )

    def test_downsample1d(self):
        # self._test_downsample1d(
        #     (
        #         1,
        #         16,
        #         2,
        #     ),
        #     use_conv=False,
        #     tolerance=1e-4,
        # )
        self._test_downsample1d(
            (
                1,
                16,
                2,
            ),
            use_conv=True,
            tolerance=3e-4,
        )

    def test_downsample2d(self):
        for norm in [None, "ln_norm", "rms_norm"]:
            self._test_downsample2d(
                (
                    1,
                    4,
                    64,
                    64,
                ),
                use_conv=True,
                norm_type=norm,
                tolerance=1e-3,
            )
            self._test_downsample2d(
                (
                    1,
                    4,
                    64,
                    64,
                ),
                use_conv=False,
                norm_type=norm,
                tolerance=1e-3,
            )


if __name__ == "__main__":
    unittest.main()
