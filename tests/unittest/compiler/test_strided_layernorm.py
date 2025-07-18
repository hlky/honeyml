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
from typing import List

import torch
from honey.compiler import compile_model, ops
from honey.compiler.ops.common.epilogue import FuncEnum
from honey.frontend import Tensor
from honey.testing import detect_target
from honey.utils import shape_utils, torch_utils
from parameterized import param, parameterized


def build_honey_module(
    *,
    batch_sizes,
    input_nonbatch_shape,
    start_indices,
    end_indices,
    n_normalize_over_last_dims,
    gamma_is_none,
    beta_is_none,
    fuse_sigmoid_mul,
    eps,
    test_id,
    honey_dtype="float16",
    workdir="./tmp",
    test_name="strided_layernorm",
    use_welford_algorithm=False,
):
    target = detect_target(
        layernorm_use_welford_algorithm=use_welford_algorithm,
    )
    X0 = Tensor(
        shape=[
            shape_utils.gen_int_var_min_max(values=batch_sizes, name="input_batch"),
            *input_nonbatch_shape,
        ],
        dtype=honey_dtype,
        name="input",
        is_input=True,
    )
    X1 = ops.dynamic_slice()(X0, start_indices, end_indices)
    layernorm_weight_shape = X1.shape()[-n_normalize_over_last_dims:]
    if gamma_is_none:
        X2 = None
    else:
        X2 = Tensor(
            shape=layernorm_weight_shape,
            dtype=honey_dtype,
            name="gamma",
            is_input=True,
        )
    if beta_is_none:
        X3 = None
    else:
        X3 = Tensor(
            shape=layernorm_weight_shape,
            dtype=honey_dtype,
            name="beta",
            is_input=True,
        )
    if fuse_sigmoid_mul:
        layernorm_op = ops.layernorm()
        sigmoid_op = ops.elementwise(FuncEnum.SIGMOID)
        mul_op = ops.elementwise(FuncEnum.MUL)
        layernorm_out = layernorm_op(X1, X2, X3, layernorm_weight_shape, eps=eps)
        sigmoid_out = sigmoid_op(layernorm_out)
        _ = mul_op(sigmoid_out, X1)
        fused_op = ops.layernorm_sigmoid_mul(layernorm_op, sigmoid_op, mul_op)
        output = fused_op()
    else:
        output = ops.layernorm()(X1, X2, X3, layernorm_weight_shape, eps)
    output._attrs["is_output"] = True
    output._attrs["name"] = "output"
    dll_name = f"test_{test_id}.so"
    return compile_model(
        output,
        target,
        workdir,
        f"{test_name}_{test_id}",
        dll_name=dll_name,
    )


def eval_pt(
    *,
    batch_size,
    input_nonbatch_shape,
    start_indices,
    end_indices,
    n_normalize_over_last_dims,
    gamma_is_none,
    beta_is_none,
    fuse_sigmoid_mul,
    eps,
    dtype=torch.float16,
    device="cuda",
):
    dtype_device = {"dtype": dtype, "device": device}
    X0 = torch.randn(batch_size, *input_nonbatch_shape, **dtype_device)
    X1 = X0[[slice(i, j) for i, j in zip(start_indices, end_indices)]]
    layernorm_weight_shape = X1.shape[-n_normalize_over_last_dims:]
    if gamma_is_none:
        X2 = None
    else:
        X2 = torch.randn(layernorm_weight_shape, **dtype_device)
    if beta_is_none:
        X3 = None
    else:
        X3 = torch.randn(layernorm_weight_shape, **dtype_device)
    X4 = torch.nn.functional.layer_norm(
        input=X1,
        normalized_shape=layernorm_weight_shape,
        weight=X2,
        bias=X3,
        eps=eps,
    )
    if fuse_sigmoid_mul:
        output = torch.mul(X1, torch.sigmoid(X4))
    else:
        output = X4
    return {"input": X0, "gamma": X2, "beta": X3, "output": output}


class SliceLayerNormTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(SliceLayerNormTestCase, self).__init__(*args, **kwargs)
        torch.manual_seed(0)
        self._test_id = 0

    def _test_slice_layer_norm(
        self,
        *,
        input_nonbatch_shape: List[int] = (16, 64, 1024),
        n_normalize_over_last_dims: int = 1,
        batch_sizes=(3, 4, 7, 11, 18),
        gamma_is_none=False,
        beta_is_none=False,
        fuse_sigmoid_mul=False,
        eps=1e-5,
        start_indices: List[int] = (0,),
        end_indices: List[int] = (None,),
        dtype: str = "float16",
        test_name="test_slice_layer_norm",
        use_welford_algorithm=False,
    ):

        input_rank = 1 + len(input_nonbatch_shape)
        if 1 == len(start_indices) and len(start_indices) != input_rank:
            start_indices = [start_indices[0]] * input_rank
        if 1 == len(end_indices) and len(end_indices) != input_rank:
            end_indices = [end_indices[0]] * input_rank

        _layernorm_common_params = {
            "input_nonbatch_shape": input_nonbatch_shape,
            "n_normalize_over_last_dims": n_normalize_over_last_dims,
            "gamma_is_none": gamma_is_none,
            "beta_is_none": beta_is_none,
            "fuse_sigmoid_mul": fuse_sigmoid_mul,
            "eps": eps,
            "start_indices": start_indices,
            "end_indices": end_indices,
        }

        honey_module = build_honey_module(
            batch_sizes=batch_sizes,
            **_layernorm_common_params,
            test_id=self._test_id,
            honey_dtype=dtype,
            test_name=f"{test_name}_{dtype}",
            use_welford_algorithm=use_welford_algorithm,
        )
        self._test_id += 1
        pt_dtype = torch_utils.string_to_torch_dtype(dtype)
        for batch_size in batch_sizes:
            pt_tensors = eval_pt(
                batch_size=batch_size, **_layernorm_common_params, dtype=pt_dtype
            )
            honey_inputs = {
                k: v for k, v in pt_tensors.items() if v is not None and k != "output"
            }
            honey_outputs = {"output": torch.empty_like(pt_tensors["output"])}
            honey_module.run_with_tensors(honey_inputs, honey_outputs)
            torch.testing.assert_close(
                honey_outputs["output"],
                pt_tensors["output"],
                atol=1e-3,
                rtol=1e-3,
            )

    def _test_slice_layer_norm_kernels(
        self,
        **kwargs,
    ):
        for start_indices, end_indices, input_nonbatch_shape, use_welford_algorithm in (
            # (cuda-half4) kernel
            ((0, 0, 0, 4), (None, None, None, 36), (4, 1, 40), False),
            # (generic n < 1024) kernel
            ((0, 0, 0, 11), (None, None, None, 13), (4, 1, 15), False),
            # (cuda-half; block size = 512) kernel
            ((0, 0, 0, 1), (None, None, None, 1026), (4, 1, 1027), True),
        ):
            self._test_slice_layer_norm(
                start_indices=start_indices,
                end_indices=end_indices,
                input_nonbatch_shape=input_nonbatch_shape,
                use_welford_algorithm=use_welford_algorithm,
                **kwargs,
            )

    def _test_middle_slice_layer_norm_kernels(
        self,
        **kwargs,
    ):
        for start_indices, end_indices, input_nonbatch_shape, use_welford_algorithm in (
            # (cuda-half4) kernel
            ((0, 0, 4, 0), (None, None, 36, None), (2, 40, 4), False),
            # (generic n < 1024) kernel
            ((0, 0, 11, 0), (None, None, 13, None), (2, 15, 2), True),
            # (cuda-half; block size = 512) kernel
            ((0, 0, 1, 0), (None, None, 1026, None), (2, 1027, 2), False),
        ):
            self._test_slice_layer_norm(
                start_indices=start_indices,
                end_indices=end_indices,
                input_nonbatch_shape=input_nonbatch_shape,
                use_welford_algorithm=use_welford_algorithm,
                **kwargs,
            )

    @parameterized.expand(
        [
            param(0, 1, True, True),
            param(1, 1, True, False),
            param(2, 1, False, True),
            param(3, 1, False, False),
            param(4, 3, True, True),
            param(5, 3, True, False),
            param(6, 3, False, True),
            param(7, 3, False, False),
        ]
    )
    def test_slice_layer_norm_float16(
        self,
        test_id,
        n_normalize_over_last_dims,
        gamma_is_none,
        beta_is_none,
    ):
        self._test_slice_layer_norm_kernels(
            n_normalize_over_last_dims=n_normalize_over_last_dims,
            gamma_is_none=gamma_is_none,
            beta_is_none=beta_is_none,
            fuse_sigmoid_mul=False,
            test_name=f"test_slice_layer_norm_float16_{test_id}",
        )

    @parameterized.expand(
        [
            param(0, 2, True, True),
            param(1, 2, True, False),
            param(2, 2, False, True),
            param(3, 2, False, False),
            param(4, 3, True, True),
            param(5, 3, True, False),
            param(6, 3, False, True),
            param(7, 3, False, False),
        ]
    )
    def test_middle_slice_layer_norm_float16(
        self,
        test_id,
        n_normalize_over_last_dims,
        gamma_is_none,
        beta_is_none,
    ):
        self._test_middle_slice_layer_norm_kernels(
            n_normalize_over_last_dims=n_normalize_over_last_dims,
            gamma_is_none=gamma_is_none,
            beta_is_none=beta_is_none,
            fuse_sigmoid_mul=False,
            test_name=f"test_middle_slice_layer_norm_float16_{test_id}",
        )

    @parameterized.expand(
        [
            param(0, 1, True, True),
            param(1, 1, True, False),
            param(2, 1, False, True),
            param(3, 1, False, False),
            param(4, 3, True, True),
            param(5, 3, True, False),
            param(6, 3, False, True),
            param(7, 3, False, False),
        ]
    )
    def test_slice_layer_norm_fuse_sigmoid_mul_float16(
        self,
        test_id,
        n_normalize_over_last_dims,
        gamma_is_none,
        beta_is_none,
    ):
        self._test_slice_layer_norm_kernels(
            n_normalize_over_last_dims=n_normalize_over_last_dims,
            gamma_is_none=gamma_is_none,
            beta_is_none=beta_is_none,
            fuse_sigmoid_mul=True,
            test_name=f"test_slice_layer_norm_fuse_sigmoid_mul_float16_{test_id}",
        )

    @parameterized.expand(
        [
            param(0, 2, True, True),
            param(1, 2, True, False),
            param(2, 2, False, True),
            param(3, 2, False, False),
            param(4, 3, True, True),
            param(5, 3, True, False),
            param(6, 3, False, True),
            param(7, 3, False, False),
        ]
    )
    def test_middle_slice_layer_norm_fuse_sigmoid_mul_float16(
        self,
        test_id,
        n_normalize_over_last_dims,
        gamma_is_none,
        beta_is_none,
    ):
        self._test_middle_slice_layer_norm_kernels(
            n_normalize_over_last_dims=n_normalize_over_last_dims,
            gamma_is_none=gamma_is_none,
            beta_is_none=beta_is_none,
            fuse_sigmoid_mul=True,
            test_name=f"test_middle_slice_layer_norm_fuse_sigmoid_mul_float16_{test_id}",
        )

    @unittest.skipIf(
        detect_target().name() != "cuda", "fp32 is only supported in CUDA backend"
    )
    @parameterized.expand(
        [
            param(0, 1, True, True, False),
            param(1, 2, True, False, False),
            param(2, 3, False, True, True),
            param(3, 2, False, False, True),
        ]
    )
    def test_slice_layer_norm_float32(
        self,
        test_id,
        n_normalize_over_last_dims,
        gamma_is_none,
        beta_is_none,
        fuse_sigmoid_mul,
    ):
        self._test_slice_layer_norm_kernels(
            n_normalize_over_last_dims=n_normalize_over_last_dims,
            gamma_is_none=gamma_is_none,
            beta_is_none=beta_is_none,
            fuse_sigmoid_mul=fuse_sigmoid_mul,
            dtype="float32",
            test_name=f"test_slice_layer_norm_float32_{test_id}",
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
