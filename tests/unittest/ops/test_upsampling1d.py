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


_DEFAULT_BATCH_SIZE = [1, 3]


class Upsampling1dTestCase(unittest.TestCase):
    def _test_single_op(
        self,
        scale_factor=2.0,
        mode="linear",
        align_corners=False,
        batch_size=_DEFAULT_BATCH_SIZE,
        test_name="linear_upsampling1d_fp16",
        dtype="float16",
    ):
        channels = 1024
        WW = 8
        target = detect_target()
        X = Tensor(
            shape=[IntVar(values=batch_size, name="input_batch"), WW, channels],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        OP = nn.Upsampling1d(
            scale_factor=scale_factor, mode=mode, align_corners=align_corners
        )
        Y = OP(X)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", test_name)

        for b in batch_size:
            X_pt = get_random_torch_tensor([b, channels, WW], dtype=dtype)
            Y_pt = torch.nn.functional.interpolate(
                X_pt, scale_factor=scale_factor, mode=mode, align_corners=align_corners
            )
            x = torch.permute(X_pt, (0, 2, 1)).contiguous()
            y = torch.empty_like(Y_pt).permute((0, 2, 1)).contiguous()
            module.run_with_tensors([x], [y])
            y_transpose = torch.permute(y, (0, 2, 1))
            torch.testing.assert_close(
                y_transpose,
                Y_pt.to(y.dtype),
                rtol=1e-3,
                atol=1e-3,
                msg=lambda msg: f"{msg}\n\n{test_name}\npt ({Y_pt.shape}):\n{Y_pt}\n\ndinoml ({y_transpose.shape}):\n{y_transpose}\n\n",
            )

    @parameterized.expand(
        **filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16"), ("float32")],
                TestEnv.CUDA_SM80: [("bfloat16")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_upsampling1d_constructor(self, dinoml_dtype):
        if dinoml_dtype != "bfloat16":
            self._test_single_op(
                scale_factor=3.5,
                mode="linear",
                test_name=f"linear_upsampling1d_{dinoml_dtype}",
                dtype=dinoml_dtype,
            )
            self._test_single_op(
                scale_factor=2.0,
                mode="linear",
                align_corners=True,
                test_name=f"linear_align_corners_upsampling1d_{dinoml_dtype}",
                dtype=dinoml_dtype,
            )
        self._test_single_op(
            scale_factor=2.0,
            mode="nearest-exact",
            align_corners=None,
            test_name=f"nearest-exact_upsampling1d_{dinoml_dtype}",
            dtype=dinoml_dtype,
        )
        self._test_single_op(
            scale_factor=2.0,
            mode="nearest",
            align_corners=None,
            test_name=f"nearest_upsampling1d_{dinoml_dtype}",
            dtype=dinoml_dtype,
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
