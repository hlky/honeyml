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
from honey.frontend import Tensor
from honey.testing import detect_target
from honey.testing.test_utils import (
    env_variables,
    filter_test_cases_by_test_env,
    get_random_torch_tensor,
    get_torch_empty_tensor,
)


_TOLERANCE_LIMITS = {
    "float16": {"atol": 1e-1, "rtol": 1e-1},
    "float32": {"atol": 1e-1, "rtol": 1e-1},
    "bfloat16": {"atol": 3e-1, "rtol": 3e-1},
}


class GEMMBiasSigmoidTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(GEMMBiasSigmoidTestCase, self).__init__(*args, **kwargs)
        self._test_id = 0

    def _test_gemm_rcr_bias_sigmoid(
        self,
        M=128,
        K=1024,
        N=64,
        dtype="float16",
        test_suffix=None,
    ):
        X = Tensor(shape=[M, K], dtype=dtype, name="input_0", is_input=True)
        W = Tensor(shape=[N, K], dtype=dtype, name="input_1", is_input=True)
        B = Tensor(shape=[N], dtype=dtype, name="input_2", is_input=True)
        OP = ops.gemm_rcr_bias_sigmoid()
        Y = OP(X, W, B)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        if test_suffix is None:
            test_suffix = dtype
        test_name = f"gemm_rcr_bias_sigmoid_{test_suffix}_{self._test_id}"
        self._test_id += 1
        module = compile_model(Y, detect_target(), "./tmp", test_name)
        X_pt = get_random_torch_tensor([M, K], dtype)
        W_pt = get_random_torch_tensor([N, K], dtype)
        B_pt = get_random_torch_tensor([N], dtype)
        Y_pt = torch.nn.functional.linear(X_pt, W_pt, bias=B_pt)
        Y_pt = torch.sigmoid(Y_pt)

        inputs = {"input_0": X_pt, "input_1": W_pt, "input_2": B_pt}
        y = get_torch_empty_tensor([M, N], dtype)
        module.run_with_tensors(inputs, [y])
        torch.testing.assert_close(Y_pt, y, **_TOLERANCE_LIMITS[dtype])

    def test_gemm_rcr_bias_sigmoid_fp16(self):
        self._test_gemm_rcr_bias_sigmoid(dtype="float16")

    def test_gemm_rcr_bias_sigmoid_fp16_rocm(self):
        self._test_gemm_rcr_bias_sigmoid(dtype="float16")

    def test_gemm_rcr_bias_sigmoid_fp32_sm80(self):
        self._test_gemm_rcr_bias_sigmoid(dtype="float32")

    def test_gemm_rcr_bias_sigmoid_bf16(self):
        self._test_gemm_rcr_bias_sigmoid(dtype="bfloat16")

    def test_gemm_rcr_bias_sigmoid_sm90(self):
        with env_variables(
            Honey_FORCE_CUTLASS_SM90_KERNELS="1",
            INSIDE_RE_WORKER="1",
        ):
            with self.assertRaisesRegex(
                expected_exception=RuntimeError,
                expected_regex="No GEMM op instances are left after filtering",
            ):
                # input alignment < 8 not supported by SM90 kernels
                # use alignment 4 to avoid auto-padding to 8
                self._test_gemm_rcr_bias_sigmoid(
                    K=1020,
                    dtype="float16",
                    test_suffix="wrong_input_alignment_sm90",
                )

            with self.assertRaisesRegex(
                expected_exception=RuntimeError,
                expected_regex="No GEMM op instances are left after filtering",
            ):
                # output alignment < 8 not supported by SM90 TMA epilogues
                self._test_gemm_rcr_bias_sigmoid(
                    N=63,
                    dtype="float16",
                    test_suffix="wrong_output_alignment_sm90",
                )

            self._test_gemm_rcr_bias_sigmoid(
                dtype="float16",
                test_suffix="float16_force_sm90",
            )
            self._test_gemm_rcr_bias_sigmoid(
                dtype="bfloat16",
                test_suffix="bfloat16_force_sm90",
            )


filter_test_cases_by_test_env(GEMMBiasSigmoidTestCase)


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
