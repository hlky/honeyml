import unittest

import torch

from dinoml.compiler import compile_model
from dinoml.frontend import IntVar, nn, Tensor
from dinoml.testing import detect_target
from dinoml.testing.test_utils import (
    filter_test_cases_by_params,
    get_random_torch_tensor,
    TestEnv,
)
from parameterized import parameterized


_DEFAULT_BATCH_SIZE = [1, 2]
TOLERANCES = {"bfloat16": 2e-2, "float16": 1e-3, "float32": 1e-5}


class Upsampling3DAddTestCase(unittest.TestCase):
    def _test_single_op(
        self,
        scale_factor=2.0,
        mode="trilinear",
        align_corners=False,
        batch_size=_DEFAULT_BATCH_SIZE,
        test_name="trilinear_upsampling3d_add",
        dtype="float16",
    ):
        channels = 64
        F, H, W = 4, 8, 8
        target = detect_target()

        X = Tensor(
            shape=[IntVar(values=batch_size, name="batch"), F, H, W, channels],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        RES = Tensor(
            shape=[
                IntVar(values=batch_size, name="batch"),
                F * int(scale_factor),
                H * int(scale_factor),
                W * int(scale_factor),
                channels,
            ],
            dtype=dtype,
            name="residual",
            is_input=True,
        )

        OP = nn.Upsampling3d(
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )

        Y = OP(X, RES)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        module = compile_model(Y, target, "./tmp", test_name)

        for b in batch_size:
            X_pt = get_random_torch_tensor([b, channels, F, H, W], dtype=dtype)
            R_pt = get_random_torch_tensor(
                [
                    b,
                    channels,
                    int(F * scale_factor),
                    int(H * scale_factor),
                    int(W * scale_factor),
                ],
                dtype=dtype,
            )

            Y_pt = torch.nn.functional.interpolate(
                X_pt,
                scale_factor=scale_factor,
                mode=mode,
                align_corners=align_corners if mode == "trilinear" else None,
            )
            Y_pt = Y_pt + R_pt

            x = X_pt.permute(0, 2, 3, 4, 1).contiguous()
            r = R_pt.permute(0, 2, 3, 4, 1).contiguous()
            y = torch.empty_like(Y_pt).permute(0, 2, 3, 4, 1).contiguous()

            module.run_with_tensors([x, r], [y])

            y_out = y.permute(0, 4, 1, 2, 3)

            torch.testing.assert_close(
                y_out,
                Y_pt,
                rtol=TOLERANCES[dtype],
                atol=TOLERANCES[dtype],
                msg=(f"{test_name}\nExpected:\n{Y_pt}\n\nGot:\n{y_out}"),
            )

    @parameterized.expand(
        **filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16",)],
                TestEnv.CUDA_SM80: [("bfloat16",)],
                TestEnv.ROCM: [("float16",)],
            }
        )
    )
    def test_upsampling3d_add(self, dtype):
        self._test_single_op(
            scale_factor=2.0,
            mode="trilinear",
            dtype=dtype,
            test_name=f"trilinear_upsampling3d_add_{dtype}",
        )
        self._test_single_op(
            scale_factor=2.0,
            mode="trilinear",
            dtype=dtype,
            align_corners=True,
            test_name=f"trilinear_upsampling3d_align_corners_add_{dtype}",
        )

        self._test_single_op(
            scale_factor=2.0,
            mode="nearest",
            dtype=dtype,
            test_name=f"nearest_upsampling3d_add_{dtype}",
        )

        self._test_single_op(
            scale_factor=2.0,
            mode="nearest-exact",
            dtype=dtype,
            test_name=f"nearest_exact_upsampling3d_add_{dtype}",
        )
