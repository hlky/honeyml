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
    filter_test_cases_by_test_env,
    get_random_torch_tensor,
    get_torch_empty_tensor,
    has_op,
)
from honey.utils import graph_utils, shape_utils


@unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
class GEMMReshapePermuteTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)

    def _test_rcr_0213(
        self,
        ms,
        k,
        n,
        shape,
        test_name,
        dtype="float16",
        has_bias=False,
        layout="0213",
        should_fuse=True,
    ):
        target = detect_target()
        X = Tensor(
            shape=[shape_utils.gen_int_var_min_max(ms), k],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        W = Tensor(shape=[n, k], dtype=dtype, name="input_1", is_input=True)
        # B = Tensor(shape=[n], dtype="float16", name="input_2", is_input=True)
        t1, t2 = shape

        if has_bias:
            return
        else:
            m_d1 = [m // t1 for m in ms]
            Y0 = ops.gemm_rcr()(X, W)
            Y1 = ops.reshape()(
                Y0, [shape_utils.gen_int_var_min_max(m_d1), t1, t2, n // t2]
            )
            Y = ops.permute()(Y1, [0, 2, 1, 3])
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", f"gemm_rcr_{test_name}")

        if should_fuse:
            sorted_ops = graph_utils.get_sorted_ops(module.debug_sorted_graph)
            assert has_op(sorted_ops, "gemm_rcr_permute")
        else:
            return

        for m in ms:
            X_pt = get_random_torch_tensor([m, k], dtype)
            W_pt = get_random_torch_tensor([n, k], dtype)
            B_pt = get_random_torch_tensor([n], dtype)

            def torch_f(x, w, b, has_bias, shape):
                if has_bias:
                    Y_l = torch.nn.functional.linear(x, w, b)
                else:
                    Y_l = torch.nn.functional.linear(x, w)
                t1, t2 = shape
                Y_r = Y_l.reshape(m // t1, t1, t2, n // t2)
                Y_pt = torch.permute(Y_r, [0, 2, 1, 3])
                Y_out = Y_pt.reshape([m // t1, t2, -1])
                return Y_pt, Y_out

            Y_pt, _ = torch_f(X_pt, W_pt, B_pt, has_bias, shape)

            inputs = {"input_0": X_pt, "input_1": W_pt}
            if has_bias:
                inputs["input_2"] = B_pt
            y = get_torch_empty_tensor(Y_pt.shape, dtype)
            module.run_with_tensors(inputs, [y])
            self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

            # module.benchmark_with_tensors(inputs, [y], count=1000)
            # from honey.testing.benchmark_pt import benchmark_torch_function

            # t = benchmark_torch_function(
            #     1000, torch_f, X_pt, W_pt, B_pt, has_bias, shape
            # )
            # print(f"pt: {t} ms/iter")

    def test_rcr_0213_sm80(self):
        self._test_rcr_0213(
            [54],
            256,
            40000,
            [54, 10000],
            "permute_0213_1",
            has_bias=False,
            layout="0213",
        )
        self._test_rcr_0213(
            [29, 29 * 8],
            256,
            3000,
            [29, 1000],
            "permute_0213_2",
            has_bias=False,
            layout="0213",
            should_fuse=False,
        )

    def test_rcr_0213_float_sm80(self):
        self._test_rcr_0213(
            [29, 29 * 8],
            256,
            3000,
            [29, 1000],
            "permute_0213_float_2",
            dtype="float",
            has_bias=False,
            layout="0213",
            should_fuse=False,
        )


filter_test_cases_by_test_env(GEMMReshapePermuteTestCase)

if __name__ == "__main__":
    unittest.main()
