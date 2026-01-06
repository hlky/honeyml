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
import logging
import unittest
import torch

from dinoml.compiler import compile_model, ops
from dinoml.frontend import Tensor
from dinoml.testing import detect_target
from dinoml.testing.test_utils import get_random_torch_tensor


_LOGGER = logging.getLogger(__name__)


def mark_output(y):
    if type(y) is not tuple:
        y = (y,)
    for i in range(len(y)):
        y[i]._attrs["is_output"] = True
        y[i]._attrs["name"] = "output_%d" % (i)
        y_shape = [d._attrs["values"][0] for d in y[i]._attrs["shape"]]
        print(f"output_{i} shape: {y_shape}")


@unittest.skipIf(detect_target()._arch == "75", "DualGemm not supported on sm75.")
class DUALGEMMBiasTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._test_id = 0

    def _test_dual_gemm_bias(
        self,
        M=4096,
        N=4096,
        K=8192,
        benchmark=False,
        broadcast_b1=False,
        test_name="dual_gemm_bias",
        dtype="float16",
    ):
        W_shape = [1, K] if broadcast_b1 else [N, K]
        target = detect_target(use_fp16_acc=False)
        X = Tensor(
            shape=[M, K],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        W0 = Tensor(
            shape=[N, K],
            dtype=dtype,
            name="input_1",
            is_input=True,
        )
        W1 = Tensor(
            shape=W_shape,
            dtype=dtype,
            name="input_2",
            is_input=True,
        )
        Bias0 = Tensor(
            shape=[N],
            dtype=dtype,
            name="input_3",
            is_input=True,
        )
        Bias1 = Tensor(
            shape=[N],
            dtype=dtype,
            name="input_4",
            is_input=True,
        )
        OP = ops.dual_gemm_rcr_bias_fast_gelu()
        Y = OP(X, W0, W1, Bias0, Bias1)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", f"{test_name}_{self._test_id}")
        self._test_id += 1
        X_pt = get_random_torch_tensor([M, K], dtype=dtype) * 0.01
        W0_pt = get_random_torch_tensor([N, K], dtype=dtype)
        W1_pt = get_random_torch_tensor(W_shape, dtype=dtype)
        Bias0_pt = get_random_torch_tensor([N], dtype=dtype)
        Bias1_pt = get_random_torch_tensor([N], dtype=dtype)

        def pt_func(X_pt, W0_pt, W1_pt, Bias0_pt, Bias1_pt):
            Y_pt1 = torch.nn.functional.linear(X_pt, W0_pt, bias=Bias0_pt)
            Y_pt2 = torch.nn.functional.linear(X_pt, W1_pt, bias=Bias1_pt)
            gelu_act = torch.nn.functional.gelu
            Y_pt = gelu_act(Y_pt1) * Y_pt2
            return Y_pt

        Y_pt = pt_func(X_pt, W0_pt, W1_pt, Bias0_pt, Bias1_pt)

        inputs = {
            "input_0": X_pt,
            "input_1": W0_pt,
            "input_2": W1_pt,
            "input_3": Bias0_pt,
            "input_4": Bias1_pt,
        }
        y = torch.empty_like(Y_pt)
        module.run_with_tensors(inputs, [y])

        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-1, rtol=1e-1))

        if benchmark:
            # Warm up.
            for _ in range(5):
                module.run_with_tensors(inputs, [y])
            # Benchmark DinoML
            time_per_iter_ms, time_std, _ = module.benchmark_with_tensors(
                inputs,
                [y],
                count=100,
            )
            _LOGGER.info(f"DinoML GEMMxGEMM time: {time_per_iter_ms:.5f}ms")
            # Benchmark PT
            from dinoml.testing.benchmark_pt import benchmark_torch_function

            func = pt_func
            args = (X_pt, W0_pt, W1_pt, Bias0_pt, Bias1_pt)
            duration = benchmark_torch_function(100, func, *args)
            _LOGGER.info(f"PT GEMMxGEMM Time: {duration:.5f}ms")

    def test_dual_gemm_bias_fast_gelu_fp16(self):
        self._test_dual_gemm_bias(
            M=128,
            N=128,
            K=256,
            broadcast_b1=False,
            test_name="dual_gemm_bias_fast_gelu_fp16_1",
            dtype="float16",
            benchmark=True,
        )
        self._test_dual_gemm_bias(
            M=1024,
            N=1024,
            K=2048,
            broadcast_b1=False,
            test_name="dual_gemm_bias_fast_gelu_fp16_2",
            dtype="float16",
            benchmark=True,
        )
        self._test_dual_gemm_bias(
            M=4096,
            N=4096,
            K=8192,
            broadcast_b1=False,
            test_name="dual_gemm_bias_fast_gelu_fp16_3",
            dtype="float16",
            benchmark=True,
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
