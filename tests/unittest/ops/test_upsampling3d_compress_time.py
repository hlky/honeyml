import unittest

import torch

from dinoml.compiler import compile_model, ops
from dinoml.frontend import IntVar, nn, Tensor
from dinoml.testing import detect_target
from dinoml.testing.test_utils import (
    filter_test_cases_by_params,
    get_random_torch_tensor,
    TestEnv,
)
from parameterized import parameterized


_DEFAULT_BATCH_SIZE = [1, 2]
_FRAME_SIZE = [1, 3, 8]
TOLERANCES = {"bfloat16": 2e-2, "float16": 1e-3, "float32": 1e-5}


def compress_time(inputs: torch.Tensor):
    if inputs.shape[2] > 1 and inputs.shape[2] % 2 == 1:
        # split first frame
        x_first, x_rest = inputs[:, :, 0], inputs[:, :, 1:]

        x_first = torch.nn.functional.interpolate(x_first, scale_factor=2.0)
        x_rest = torch.nn.functional.interpolate(x_rest, scale_factor=2.0)
        x_first = x_first[:, :, None, :, :]
        inputs = torch.cat([x_first, x_rest], dim=2)
    elif inputs.shape[2] > 1:
        inputs = torch.nn.functional.interpolate(inputs, scale_factor=2.0)
    else:
        inputs = inputs.squeeze(2)
        inputs = torch.nn.functional.interpolate(inputs, scale_factor=2.0)
        inputs = inputs[:, :, None, :, :]
    return inputs


class Upsampling3DCompressTimeTestCase(unittest.TestCase):
    def _test_single_op(
        self,
        scale_factor=2.0,
        batch_size=_DEFAULT_BATCH_SIZE,
        test_name="trilinear_upsampling3d",
        dtype="float16",
    ):
        channels = 64
        H, W = 8, 8
        target = detect_target()

        X = Tensor(
            shape=[
                IntVar(values=batch_size, name="batch"),
                IntVar([1, 8]),
                IntVar([4, 16]),
                IntVar([4, 16]),
                channels,
            ],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )

        OP = ops.upsampling3d_compress_time()

        Y = OP(X)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True

        module = compile_model(Y, target, "./tmp", test_name)

        for b in batch_size:
            for F in _FRAME_SIZE:
                X_pt = get_random_torch_tensor([b, channels, F, H, W], dtype=dtype)

                Y_pt = compress_time(X_pt)

                x = X_pt.permute(0, 2, 3, 4, 1).contiguous()
                y = torch.empty_like(Y_pt).permute(0, 2, 3, 4, 1).contiguous()

                module.run_with_tensors([x], [y])

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
    def test_upsampling3d(self, dtype):
        self._test_single_op(
            scale_factor=2.0,
            dtype=dtype,
            test_name=f"nearest_upsampling3d_compress_time_{dtype}",
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
