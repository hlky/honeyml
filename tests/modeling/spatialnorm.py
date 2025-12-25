import unittest

from typing import cast, List, Optional, Tuple

import torch
from honey.compiler import compile_model, ops
from honey.frontend import Tensor
from honey.testing import detect_target
from honey.testing.test_utils import get_random_torch_tensor
from honey.modeling.diffusers.attention_processor import SpatialNorm
from honey.builder.config import mark_output

from diffusers.models.attention_processor import SpatialNorm as SpatialNorm_torch


class SpatialNormTestCase(unittest.TestCase):
    def _test_spatial_norm(
        self,
        f_shape: List[int],
        zq_shape: List[int],
        f_channels: int,
        zq_channels: int,
        dtype: str = "float16",
        tolerance: float = 1e-5,
    ):
        f = get_random_torch_tensor(f_shape, dtype=dtype)
        zq = get_random_torch_tensor(zq_shape, dtype=dtype)
        f_honey = f.clone().permute(0, 2, 3, 1).contiguous().to(f.device, f.dtype)
        zq_honey = zq.clone().permute(0, 2, 3, 1).contiguous().to(zq.device, zq.dtype)

        op = (
            SpatialNorm_torch(f_channels=f_channels, zq_channels=zq_channels)
            .eval()
            .to(f.device, f.dtype)
        )

        state_dict_pt = cast(dict[str, torch.Tensor], op.state_dict())
        state_dict_honey = {}
        for key, value in state_dict_pt.items():
            key_honey = key.replace(".", "_")
            if value.ndim == 4 and "weight" in key:
                value = value.permute(0, 2, 3, 1).contiguous()
            value = value.to(f.device, f.dtype)
            state_dict_honey[key_honey] = value

        with torch.inference_mode():
            y_pt = op.forward(f, zq)

        y = torch.empty_like(y_pt.permute(0, 2, 3, 1).contiguous()).to(
            f.device, f.dtype
        )

        F = Tensor(
            shape=[f_shape[0], f_shape[2], f_shape[3], f_shape[1]],
            dtype=dtype,
            name="F",
            is_input=True,
        )
        ZQ = Tensor(
            shape=[zq_shape[0], zq_shape[2], zq_shape[3], zq_shape[1]],
            dtype=dtype,
            name="ZQ",
            is_input=True,
        )

        op_honey = SpatialNorm(
            f_channels=f_channels, zq_channels=zq_channels, dtype=dtype
        )
        op_honey.name_parameter_tensor()
        Y = op_honey.forward(F, ZQ)
        Y = mark_output(Y, "Y")

        target = detect_target()
        test_name = (
            f"test_spatial_norm_{dtype}_f_channels{f_channels}_zq_channels{zq_channels}"
        )
        inputs = {"F": f_honey, "ZQ": zq_honey}
        module = compile_model(
            Y,
            target,
            "./tmp",
            test_name,
            constants=state_dict_honey,
        )
        module.run_with_tensors(inputs, [y])
        y = y.permute(0, 3, 1, 2).contiguous()
        torch.testing.assert_close(
            y,
            y_pt.to(y.dtype),
            rtol=tolerance,
            atol=tolerance,
            msg=lambda msg: f"{msg}\n\npt ({y_pt.shape}):\n{y_pt}\n\nhoney ({y.shape}):\n{y}\n\n",
        )

    def test_spatial_norm(self):
        self._test_spatial_norm(
            f_shape=[1, 64, 32, 32],
            zq_shape=[1, 128, 16, 16],
            f_channels=64,
            zq_channels=128,
            tolerance=3e-3,
            dtype="float16",
        )


if __name__ == "__main__":
    unittest.main()
