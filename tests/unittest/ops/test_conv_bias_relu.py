#  Copyright 2025 hlky. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import unittest

import torch

from honey.compiler import compile_model, ops
from honey.frontend import IntImm, Tensor
from honey.testing import detect_target
from honey.testing.test_utils import (
    filter_test_cases_by_params,
    get_random_torch_tensor,
    TestEnv,
)

from parameterized import parameterized


class ConvBiasReluTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)

    def _test_conv_bias_relu(
        self,
        batch=4,
        copy_op=False,
        test_name="conv2d_bias_relu",
        dtype="float16",
    ):
        target = detect_target()
        X = Tensor(
            shape=[IntImm(batch), 28, 28, 128],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        W = Tensor(
            shape=[256, 3, 3, 128],
            dtype=dtype,
            name="input_1",
            is_input=True,
        )
        B = Tensor(
            shape=[256],
            dtype=dtype,
            name="input_2",
            is_input=True,
        )
        OP = ops.conv2d(stride=1, pad=1, dilate=1, bias=True, activation="relu")
        if copy_op:
            OP = ops.conv2d(**OP._get_op_attributes())
        Y = OP(X, W, B)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", test_name)

        X_pt = get_random_torch_tensor([batch, 128, 28, 28], dtype=dtype)
        W_pt = get_random_torch_tensor([256, 128, 3, 3], dtype=dtype)
        B_pt = get_random_torch_tensor([1, 256, 1, 1], dtype=dtype)
        Y_pt = torch.nn.functional.conv2d(X_pt.float(), W_pt.float(), padding=1).to(
            dtype=X_pt.dtype
        )
        Y_pt = Y_pt + B_pt
        Y_pt = torch.nn.functional.relu(Y_pt)
        x = X_pt.permute((0, 2, 3, 1)).contiguous()
        w = W_pt.permute((0, 2, 3, 1)).contiguous()
        inputs = {"input_0": x, "input_1": w, "input_2": B_pt.squeeze()}
        y = torch.empty_like(Y_pt).permute((0, 2, 3, 1)).contiguous()
        module.run_with_tensors(inputs, [y])
        y_transpose = y.permute((0, 3, 1, 2))
        if target.name() == "cuda":
            if dtype == "float32":
                torch.testing.assert_close(Y_pt, y_transpose, atol=5e-2, rtol=1e-2)
            elif dtype == "float16":
                torch.testing.assert_close(Y_pt, y_transpose, atol=1e-2, rtol=1e-2)
            elif dtype == "bfloat16":
                torch.testing.assert_close(Y_pt, y_transpose, atol=2e-2, rtol=1e-2)
        else:
            torch.testing.assert_close(Y_pt, y_transpose, atol=1.25e-1, rtol=1e-1)

    @parameterized.expand(
        **filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16")],
                TestEnv.CUDA_SM80: [("bfloat16"), ("float32")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_conv2d_bias_relu(self, dtype):
        self._test_conv_bias_relu(
            test_name=f"conv2d_bias_relu_{dtype}",
            dtype=dtype,
        )
        self._test_conv_bias_relu(
            copy_op=True,
            test_name=f"conv2d_bias_relu_{dtype}_copy_op",
            dtype=dtype,
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
