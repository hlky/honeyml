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
"""
Unittests for fused_elementwise Operator.
"""

import itertools
import unittest
from typing import List

import torch

from honey.compiler import compile_model, ops, transform
from honey.compiler.base import IntImm
from honey.compiler.ops.common.epilogue import FuncEnum
from honey.compiler.stable_set import StableSet
from honey.compiler.transform.fuse_ops import _get_inputs_outputs
from honey.frontend import Tensor
from honey.testing import detect_target
from honey.testing.test_utils import (
    filter_test_cases_by_params,
    filter_test_cases_by_test_env,
    get_random_torch_tensor,
    get_torch_empty_tensor,
    get_torch_full_tensor,
    TestEnv,
)
from honey.utils import shape_utils
from parameterized import parameterized


_Honey_DTYPE_TO_PYTORCH_DTYPE = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


class FusedElementwiseTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)

    def _test_fused_elementwise_constructor(self, honey_dtype):
        BATCH_SIZE = 1024
        M = 256
        K = 128

        op1 = ops.elementwise(FuncEnum.ADD)
        op1._attrs["name"] = "e1"
        op2 = ops.elementwise(FuncEnum.TANH)
        op2._attrs["name"] = "e2"
        X1 = Tensor(
            shape=[BATCH_SIZE, M, K],
            dtype=honey_dtype,
            name="input0",
            is_input=True,
        )
        X2 = Tensor(
            shape=[],
            dtype=honey_dtype,
            name="X2",
            value=3.0,
        )
        X3 = op1(X1, X2)
        X3._attrs["name"] = "X3"
        X4 = op2(X3)
        X4._attrs["name"] = "output0"
        X4._attrs["is_output"] = True

        graph = transform.toposort(X4)
        transform.name_graph(graph)
        transform.mark_param_tensor(graph)
        transform.refine_graph(graph)
        inputs, outputs, external_inputs, external_outputs = _get_inputs_outputs(
            {op1, op2}, {op1, op2}
        )
        for tensor in itertools.chain(inputs, outputs):
            tensor._attrs["src_ops"] = tensor._attrs["src_ops"] - {op1, op2}
            tensor._attrs["dst_ops"] = tensor._attrs["dst_ops"] - {op1, op2}
        fused_op = ops.fused_elementwise(
            [op1, op2],
            external_inputs,
            external_outputs,
        )
        fused_op._attrs["name"] = "fused_elementwise0"

        self.assertEqual(fused_op._attrs["inputs"], [X1])
        self.assertEqual(fused_op._attrs["outputs"], [X4])

        self.assertEqual(X4._attrs["src_ops"], StableSet({fused_op}))
        self.assertEqual(X1._attrs["dst_ops"], StableSet({fused_op}))

        self.assertEqual(fused_op._attrs["depth"], 0)
        self.assertEqual(X1._attrs["depth"], 0)
        self.assertEqual(X4._attrs["depth"], 2)

    @parameterized.expand(
        **filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16"), ("float32")],
                TestEnv.CUDA_SM80: [("bfloat16")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_fused_elementwise_constructor(self, honey_dtype):
        self._test_fused_elementwise_constructor(honey_dtype)

    def _test_fused_elementwise_e2e(
        self,
        batch_sizes,
        ms,
        ks,
        test_name,
        honey_dtype,
        use_fp32_acc=False,
    ):
        torch_dtype = _Honey_DTYPE_TO_PYTORCH_DTYPE[honey_dtype]
        X1 = Tensor(
            shape=[
                shape_utils.gen_int_var_min_max(batch_sizes),
                shape_utils.gen_int_var_min_max(ms),
                shape_utils.gen_int_var_min_max(ks),
            ],
            dtype=honey_dtype,
            name="input0",
            is_input=True,
        )
        X2 = Tensor(
            shape=[],
            dtype=honey_dtype,
            name="X2",
            value=3.0,
        )
        X3 = X1 + X2
        X3._attrs["name"] = "X3"
        X4 = ops.tanh(X3)
        X4._attrs["name"] = "output0"
        X4._attrs["is_output"] = True

        target = detect_target(elementwise_use_fp32_acc=use_fp32_acc)
        module = compile_model(
            X4,
            target,
            "./tmp",
            f"fused_elementwise_{test_name}_{use_fp32_acc}",
        )

        for batch_size in batch_sizes:
            for m in ms:
                for k in ks:
                    x1_pt = torch.randn(batch_size, m, k).cuda().to(dtype=torch_dtype)
                    x4_pt = torch.tanh(x1_pt + 3.0)

                    x4 = torch.empty([batch_size, m, k]).cuda().to(dtype=torch_dtype)
                    module.run_with_tensors([x1_pt], [x4])
                    self.assertTrue(torch.allclose(x4, x4_pt, atol=1e-2, rtol=1e-2))

    @parameterized.expand(
        **filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16"), ("float32")],
                TestEnv.CUDA_SM80: [("bfloat16")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_fused_elementwise_e2e(self, honey_dtype):
        self._test_fused_elementwise_e2e(
            batch_sizes=[1024],
            ms=[256],
            ks=[128],
            test_name=f"static_shapes_{honey_dtype}",
            honey_dtype=honey_dtype,
        )
        self._test_fused_elementwise_e2e(
            batch_sizes=[1, 99, 998, 1024],
            ms=[256],
            ks=[128],
            test_name=f"dynamic_batch_size_{honey_dtype}",
            honey_dtype=honey_dtype,
        )
        self._test_fused_elementwise_e2e(
            batch_sizes=[1024],
            ms=[1, 128, 256],
            ks=[128],
            test_name=f"dynamic_m_{honey_dtype}",
            honey_dtype=honey_dtype,
        )
        self._test_fused_elementwise_e2e(
            batch_sizes=[1024],
            ms=[256],
            ks=[1, 3, 8, 128],
            test_name=f"dynamic_k_{honey_dtype}",
            honey_dtype=honey_dtype,
        )
        for use_fp32_acc in (False, True):
            self._test_fused_elementwise_e2e(
                batch_sizes=[700, 80, 1024],
                ms=[23, 78, 256],
                ks=[10, 30, 128],
                test_name=f"dynamic_all_{honey_dtype}",
                honey_dtype=honey_dtype,
                use_fp32_acc=use_fp32_acc,
            )

    def _test_fused_elementwise_kernel1(
        self,
        honey_dtype,
        use_fp32_acc=False,
    ):
        BATCH_SIZE = 1024
        M = 1496
        X1 = Tensor(
            shape=[IntImm(BATCH_SIZE), IntImm(2), IntImm(M)],
            dtype=honey_dtype,
            name="input0",
            is_input=True,
        )
        X2 = Tensor(
            shape=[],
            dtype=honey_dtype,
            name="constant_number",
            value=2.0,
        )
        X3 = Tensor(
            shape=[IntImm(2), IntImm(M)],
            dtype=honey_dtype,
            name="constant_matrix",
            is_input=True,
        )
        X4 = ops.elementwise(FuncEnum.SIGN)(X1)
        X5 = ops.elementwise(FuncEnum.ABS)(X1)
        X6 = ops.elementwise(FuncEnum.ADD)(X5, X2)
        X7 = ops.elementwise(FuncEnum.LOGE)(X6)
        X8 = ops.elementwise(FuncEnum.MUL)(X4, X7)
        X9 = ops.elementwise(FuncEnum.MUL)(X8, X3)
        X9._attrs["is_output"] = True
        X9._attrs["name"] = "output0"

        target = detect_target(elementwise_use_fp32_acc=use_fp32_acc)
        module = compile_model(
            X9,
            target,
            "./tmp",
            f"fused_elementwise_kernel1_{honey_dtype}_{use_fp32_acc}",
        )

        x1_pt = get_random_torch_tensor((BATCH_SIZE, 2, M), honey_dtype)
        x3_pt = get_random_torch_tensor((2, M), honey_dtype)
        x9_pt = torch.sign(x1_pt) * torch.log1p(torch.abs(x1_pt) + 1) * x3_pt

        inputs = {"input0": x1_pt, "constant_matrix": x3_pt}
        x9 = get_torch_empty_tensor([BATCH_SIZE, 2, M], honey_dtype)
        module.run_with_tensors(inputs, [x9])
        torch.testing.assert_close(x9, x9_pt, atol=1e-2, rtol=1e-2)

    @parameterized.expand(
        **filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16"), ("float32")],
                TestEnv.CUDA_SM80: [("bfloat16")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_fused_elementwise_kernel1(self, honey_dtype):
        for use_fp32_acc in (False, True):
            self._test_fused_elementwise_kernel1(
                honey_dtype=honey_dtype,
                use_fp32_acc=use_fp32_acc,
            )

    def _test_sigmoid(self, input_size, test_name, honey_dtype, use_fast_math=True):
        torch_dtype = _Honey_DTYPE_TO_PYTORCH_DTYPE[honey_dtype]
        X1 = Tensor(
            shape=[IntImm(input_size[0]), IntImm(input_size[1])],
            dtype=honey_dtype,
            name="input0",
            is_input=True,
        )
        X2 = ops.elementwise(FuncEnum.SIGMOID)(X1)
        X2._attrs["is_output"] = True
        X2._attrs["name"] = "output0"

        target = detect_target(use_fast_math=use_fast_math)
        module = compile_model(X2, target, "./tmp", test_name)

        x1_pt = (
            (torch.rand(input_size, device="cuda", dtype=torch_dtype) - 0.5) * 2.0
        ) * torch.finfo(torch_dtype).max
        x2_pt = torch.sigmoid(x1_pt)

        x2 = torch.empty_like(x2_pt)
        module.run_with_tensors([x1_pt], [x2])
        if use_fast_math:
            self.assertTrue(torch.allclose(x2, x2_pt, atol=1e-2, rtol=1e-2))
        else:
            self.assertTrue(torch.equal(x2, x2_pt), f"{x2=}\n{x2_pt=}")
        # sanity checks
        self.assertEqual(torch.sum(x2 < 0), 0)
        self.assertEqual(torch.sum(x2 > 1), 0)

    @parameterized.expand(
        **filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16"), ("float32")],
                TestEnv.CUDA_SM80: [("bfloat16")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_sigmoid(self, honey_dtype):
        self._test_sigmoid([1024, 2 * 1496], f"sigmoid_1_{honey_dtype}", honey_dtype)
        self._test_sigmoid([1024, 23744], f"sigmoid_2_{honey_dtype}", honey_dtype)
        self._test_sigmoid([1024, 70144], f"sigmoid_3_{honey_dtype}", honey_dtype)
        # use_fast_math = False
        self._test_sigmoid(
            [1024, 70144],
            f"sigmoid_no_fast_math_{honey_dtype}",
            honey_dtype,
            use_fast_math=False,
        )

    def _test_tanh(self, input_size, test_name, honey_dtype, use_fast_math=True):
        assert len(input_size) == 2
        torch_dtype = _Honey_DTYPE_TO_PYTORCH_DTYPE[honey_dtype]
        X1 = Tensor(
            shape=[IntImm(input_size[0]), IntImm(input_size[1])],
            dtype=honey_dtype,
            name="input0",
            is_input=True,
        )
        X2 = ops.elementwise(FuncEnum.TANH)(X1)
        X2._attrs["is_output"] = True
        X2._attrs["name"] = "output0"

        target = detect_target(use_fast_math=use_fast_math)
        module = compile_model(X2, target, "./tmp", test_name)

        x1_pt = torch.randn(input_size).cuda().to(dtype=torch_dtype)
        x2_pt = torch.tanh(x1_pt)

        x2 = torch.empty(input_size).cuda().to(dtype=torch_dtype)
        module.run_with_tensors([x1_pt], [x2])
        if use_fast_math:
            self.assertTrue(torch.allclose(x2, x2_pt, atol=1e-2, rtol=1e-2))
        else:
            self.assertTrue(torch.equal(x2, x2_pt))

    @parameterized.expand(
        **filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16"), ("float32")],
                # float16 device function is different for SM80 and lower
                TestEnv.CUDA_SM80: [("float16"), ("bfloat16")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_tanh(self, honey_dtype):
        self._test_tanh([1024, 22400], f"tanh_1_{honey_dtype}", honey_dtype)
        self._test_tanh([1024, 70144], f"tanh_2_{honey_dtype}", honey_dtype)
        self._test_tanh([1024, 23744], f"tanh_3_{honey_dtype}", honey_dtype)
        # use_fast_math = False
        self._test_tanh(
            [1024, 23744],
            f"tanh_no_fast_math_{honey_dtype}",
            honey_dtype,
            use_fast_math=False,
        )

    def _test_gelu(self, input_size, test_name, honey_dtype, fast_gelu=False):
        assert len(input_size) == 2
        torch_dtype = _Honey_DTYPE_TO_PYTORCH_DTYPE[honey_dtype]
        X1 = Tensor(
            shape=[IntImm(input_size[0]), IntImm(input_size[1])],
            dtype=honey_dtype,
            name="input0",
            is_input=True,
        )
        if fast_gelu:
            X2 = ops.elementwise(FuncEnum.FASTGELU)(X1)
        else:
            X2 = ops.elementwise(FuncEnum.GELU)(X1)
        X2._attrs["is_output"] = True
        X2._attrs["name"] = "output0"

        target = detect_target()
        module = compile_model(X2, target, "./tmp", test_name)

        x1_pt = torch.randn(input_size).cuda().to(dtype=torch_dtype)
        x2_pt = torch.nn.functional.gelu(x1_pt)

        x2 = torch.empty(input_size).cuda().to(dtype=torch_dtype)
        module.run_with_tensors([x1_pt], [x2])
        self.assertTrue(torch.allclose(x2, x2_pt, atol=1e-2, rtol=1e-2))

    @parameterized.expand(
        **filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16"), ("float32")],
                TestEnv.CUDA_SM80: [("bfloat16")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_gelu(self, honey_dtype):
        self._test_gelu([1024, 22400], f"gelu_1_{honey_dtype}", honey_dtype)
        self._test_gelu([1024, 70144], f"fast_gelu_1_{honey_dtype}", honey_dtype, True)

    def _test_min_max(
        self,
        input_size: List[List[int]],
        test_name: str,
        is_min: bool,
        add_nans: bool,
        honey_dtype,
    ) -> None:
        assert len(input_size) == 2
        X0 = Tensor(
            shape=[IntImm(input_size[0]), IntImm(input_size[1])],
            dtype=honey_dtype,
            name="input0",
            is_input=True,
        )
        X1 = Tensor(
            shape=[IntImm(input_size[0]), IntImm(input_size[1])],
            dtype=honey_dtype,
            name="input1",
            is_input=True,
        )
        if is_min:
            result = ops.elementwise(FuncEnum.MIN)(X0, X1)
        else:
            result = ops.elementwise(FuncEnum.MAX)(X0, X1)

        result._attrs["is_output"] = True
        result._attrs["name"] = "output0"

        target = detect_target()
        module = compile_model(result, target, "./tmp", test_name)

        x0_pt = get_random_torch_tensor(input_size, honey_dtype)
        x1_pt = get_random_torch_tensor(input_size, honey_dtype)
        if add_nans:
            x1_pt[0].fill_(float("nan"))

        if is_min:
            x2_pt = torch.min(x0_pt, x1_pt)
        else:
            x2_pt = torch.max(x0_pt, x1_pt)

        inputs = {"input0": x0_pt, "input1": x1_pt}
        x2 = get_torch_empty_tensor(input_size, honey_dtype)
        module.run_with_tensors(inputs, [x2])

        if add_nans:
            nans = get_torch_full_tensor(x2_pt[0].shape, float("nan"), honey_dtype)
            torch.testing.assert_close(nans, x2_pt[0], equal_nan=True)
            torch.testing.assert_close(nans, x2[0], equal_nan=True)

        torch.testing.assert_close(x2, x2_pt, atol=1e-2, rtol=1e-2, equal_nan=True)

    @parameterized.expand(
        **filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16"), ("float32")],
                TestEnv.CUDA_SM80: [("bfloat16")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_min(self, honey_dtype):
        self._test_min_max(
            [512, 512],
            test_name=f"min_nonan_{honey_dtype}",
            is_min=True,
            add_nans=False,
            honey_dtype=honey_dtype,
        )
        self._test_min_max(
            [512, 512],
            test_name=f"min_nan_{honey_dtype}",
            is_min=True,
            add_nans=True,
            honey_dtype=honey_dtype,
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
    def test_max(self, honey_dtype):
        self._test_min_max(
            [512, 512],
            test_name=f"max_nonan_{honey_dtype}",
            is_min=False,
            add_nans=False,
            honey_dtype=honey_dtype,
        )
        self._test_min_max(
            [512, 512],
            test_name=f"max_nan_{honey_dtype}",
            is_min=False,
            add_nans=True,
            honey_dtype=honey_dtype,
        )

    def _test_clamp(
        self,
        input_size: List[List[int]],
        min_val: int,
        max_val: int,
        test_name: str,
        honey_dtype,
    ) -> None:
        assert len(input_size) == 2 or len(input_size) == 0
        torch_dtype = _Honey_DTYPE_TO_PYTORCH_DTYPE[honey_dtype]
        X0 = Tensor(
            shape=[IntImm(input_size[0]), IntImm(input_size[1])] if input_size else [],
            dtype=honey_dtype,
            name="input0",
            is_input=True,
        )
        result = ops.clamp()(X0, min_val, max_val)
        result._attrs["is_output"] = True
        result._attrs["name"] = "output0"

        target = detect_target()
        module = compile_model(result, target, "./tmp", test_name)

        x0_pt = torch.randn(input_size).cuda().to(dtype=torch_dtype)

        x1_pt = torch.clamp(x0_pt, min_val, max_val)

        x1 = torch.empty(input_size).cuda().to(dtype=torch_dtype)
        module.run_with_tensors([x0_pt], [x1])

        self.assertTrue(torch.allclose(x1, x1_pt, atol=1e-2, rtol=1e-2))

    @parameterized.expand(
        **filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16"), ("float32")],
                TestEnv.CUDA_SM80: [("bfloat16")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_clamp(self, honey_dtype):
        self._test_clamp([512, 106], -1, 1, f"clamp_0_{honey_dtype}", honey_dtype)
        self._test_clamp([128, 46], None, 1, f"clamp_1_{honey_dtype}", honey_dtype)
        self._test_clamp([56, 265], -1, None, f"clamp_2_{honey_dtype}", honey_dtype)
        self._test_clamp([17, 123], 1, -1, f"clamp_3_{honey_dtype}", honey_dtype)
        self._test_clamp([], 1, -1, f"clamp_4_{honey_dtype}", honey_dtype)

    def _test_operator_overload(self, honey_dtype):
        input_size = [4, 2]
        torch_dtype = _Honey_DTYPE_TO_PYTORCH_DTYPE[honey_dtype]
        X1 = Tensor(
            shape=input_size,
            dtype=honey_dtype,
            name="input0",
            is_input=True,
        )
        X2 = Tensor(
            shape=input_size,
            dtype=honey_dtype,
            name="input1",
            is_input=True,
        )
        OUTPUT = -ops.tanh(X1 + X2) + ops.tanh(X2) + ops.tanh(X1)
        OUTPUT._attrs["is_output"] = True
        OUTPUT._attrs["name"] = "output"

        target = detect_target()
        module = compile_model(OUTPUT, target, "./tmp", f"test_op_overload_{honey_dtype}")

        x1_pt = torch.randn(input_size).cuda().to(dtype=torch_dtype)
        x2_pt = torch.randn(input_size).cuda().to(dtype=torch_dtype)
        output_pt = -torch.tanh(x1_pt + x2_pt) + torch.tanh(x2_pt) + torch.tanh(x1_pt)

        output = torch.empty(input_size).cuda().to(dtype=torch_dtype)
        module.run_with_tensors([x1_pt, x2_pt], [output])
        self.assertTrue(torch.allclose(output, output_pt, atol=1e-2, rtol=1e-2))

    @parameterized.expand(
        **filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16"), ("float32")],
                TestEnv.CUDA_SM80: [("bfloat16")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_operator_overload(self, honey_dtype):
        self._test_operator_overload(honey_dtype)

    def _test_operator_overload_with_constant_number(self, honey_dtype):
        input_size = [4, 2]
        torch_dtype = _Honey_DTYPE_TO_PYTORCH_DTYPE[honey_dtype]
        X1 = Tensor(
            shape=input_size,
            dtype=honey_dtype,
            name="input0",
            is_input=True,
        )
        OUTPUT = 10 / ops.tanh(X1 + 5)
        OUTPUT._attrs["is_output"] = True
        OUTPUT._attrs["name"] = "output"

        target = detect_target()
        module = compile_model(OUTPUT, target, "./tmp", f"test_op_overload_{honey_dtype}")

        x1_pt = torch.randn(input_size).cuda().to(dtype=torch_dtype)
        output_pt = 10 / torch.tanh(x1_pt + 5)
        output = torch.empty(input_size).cuda().to(dtype=torch_dtype)
        module.run_with_tensors([x1_pt], [output])
        self.assertTrue(torch.allclose(output, output_pt, atol=1e-2, rtol=1e-2))

    @parameterized.expand(
        **filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16"), ("float32")],
                TestEnv.CUDA_SM80: [("bfloat16")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_operator_overload_with_constant_number(self, honey_dtype):
        self._test_operator_overload_with_constant_number(honey_dtype)


class FusedElementwisePowerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)

    def _test_power(self, input_size, exp, test_name, honey_dtype):
        print(f"Running test {test_name} with exp = {exp}")
        assert len(input_size) == 2
        torch_dtype = _Honey_DTYPE_TO_PYTORCH_DTYPE[honey_dtype]
        X1 = Tensor(
            shape=[IntImm(input_size[0]), IntImm(input_size[1])],
            dtype=honey_dtype,
            name="input0",
            is_input=True,
        )
        X2 = ops.elementwise(FuncEnum.POW)(X1, exp)
        X2._attrs["is_output"] = True
        X2._attrs["name"] = "output0"

        target = detect_target()
        module = compile_model(X2, target, "./tmp", test_name)

        if abs(exp) < 1.0:
            x1_pt = torch.randn(input_size).cuda().to(dtype=torch_dtype) + 0.5
        else:
            x1_pt = torch.randn(input_size).cuda().to(dtype=torch_dtype)
        x2_pt = torch.pow(x1_pt, exp)

        x2 = torch.empty(input_size).cuda().to(dtype=torch_dtype)
        module.run_with_tensors([x1_pt], [x2])
        # t, _, _ = module.benchmark_with_tensors([x1_pt], [x2], count=1000)
        # bw = input_size[0] * input_size[1] * 2 * 2 / (t * 1e9 * 1e-3)
        # print(f"BW: {bw} GB/s")
        torch.testing.assert_close(x2, x2_pt, atol=1e-3, rtol=1e-3, equal_nan=True)

    @parameterized.expand(
        itertools.product(
            (0, 1, -1, 0.5, -0.5, 2, -2, 1.4, 3),
            ([1024, 1024], [1025, 1025]),
        )
    )
    def test_power_float16(self, exp, shape):
        dtype = "float16"
        self._test_power(
            shape,
            exp,
            f"pow_{shape[0]}_{shape[1]}_{exp}_{dtype}",
            dtype,
        )

    def test_power_float32_sm80(self):
        self._test_power(
            (1024, 1024),
            2.5,
            "pow_float32",
            "float32",
        )

    def test_power_bfloat16_bf16(self):
        self._test_power(
            (1024, 1024),
            1.2,
            "pow_bfloat16",
            "bfloat16",
        )


filter_test_cases_by_test_env(FusedElementwisePowerTestCase)


if __name__ == "__main__":
    unittest.main()
