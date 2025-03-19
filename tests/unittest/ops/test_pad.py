import unittest

import torch

from honey.compiler import compile_model, ops
from honey.frontend import IntVar, Tensor
from honey.testing import detect_target
from honey.testing.test_utils import (
    filter_test_cases_by_params,
    get_random_torch_tensor,
    TestEnv,
)
from parameterized import parameterized


class PadTestCase(unittest.TestCase):
    def _test_single_op(
        self,
        pad,
        mode,
        input_shape,
        value=0.0,
        test_name="pad_fp16",
        dtype="float16",
    ):
        target = detect_target()
        honey_shape = input_shape
        if len(input_shape) == 4:
            n, c, h, w = input_shape
            honey_shape = [n, h, w, c]
        X = Tensor(
            shape=honey_shape,
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        OP = ops.pad(pad=pad, mode=mode, value=value)
        Y = OP(X)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", test_name)

        X_pt = get_random_torch_tensor(input_shape, dtype=dtype)
        if mode == "constant":
            Y_pt = torch.nn.functional.pad(X_pt, pad, mode=mode, value=value)
        else:
            Y_pt = torch.nn.functional.pad(X_pt, pad, mode=mode)
        x = X_pt
        y = torch.empty_like(Y_pt)
        if len(input_shape) == 4:
            x = torch.permute(x, (0, 2, 3, 1)).contiguous()
            y = torch.permute(y, (0, 2, 3, 1)).contiguous()
        module.run_with_tensors([x], [y])
        if len(input_shape) == 4:
            y = torch.permute(y, (0, 3, 1, 2)).contiguous()
        torch.testing.assert_close(
            y,
            Y_pt.to(y.dtype),
            rtol=1e-3,
            atol=1e-3,
            msg=lambda msg: f"{msg}\n\n{test_name}\npt ({Y_pt.shape}):\n{Y_pt}\n\nhoney ({y.shape}):\n{y}\n\n",
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
    def test_pad(self, honey_dtype):
        pad_modes = ["constant", "reflect", "replicate", "circular"]
        # PyTorch only supports constant pad for 1D tensor
        for pad_mode in ["constant"]:
            for pad in [(1, 1), (0, 1), (1, 0)]:
                self._test_single_op(
                    pad=pad,
                    mode=pad_mode,
                    value=1.0,
                    input_shape=[10],
                    test_name=f"pad_{pad_mode}_1d_{honey_dtype}_{'-'.join([str(n) for n in pad])}",
                    dtype=honey_dtype,
                )
        for pad_mode in pad_modes:
            for pad in [(1, 1), (0, 1), (1, 0)]:
                self._test_single_op(
                    pad=pad,
                    mode=pad_mode,
                    value=1.0,
                    input_shape=[3, 4],
                    test_name=f"pad_{pad_mode}_2d_{honey_dtype}_{'-'.join([str(n) for n in pad])}",
                    dtype=honey_dtype,
                )
        for pad_mode in pad_modes:
            for pad in [(1, 1, 2, 2), (0, 1, 0, 1), (0, 1, 0, 0)]:
                self._test_single_op(
                    pad=pad,
                    mode=pad_mode,
                    value=1.0,
                    input_shape=[2, 3, 4],
                    test_name=f"pad_{pad_mode}_3d_{honey_dtype}_{'-'.join([str(n) for n in pad])}",
                    dtype=honey_dtype,
                )
        for pad_mode in pad_modes:
            for pad in [(1, 1, 2, 2), (0, 1, 0, 1), (0, 1, 0, 0)]:
                self._test_single_op(
                    pad=pad,
                    mode=pad_mode,
                    value=1.0,
                    input_shape=[1, 3, 3, 3],
                    test_name=f"pad_{pad_mode}_4d_{honey_dtype}_{'-'.join([str(n) for n in pad])}",
                    dtype=honey_dtype,
                )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
