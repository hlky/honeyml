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

import itertools
import json
import logging
import unittest
import uuid

import torch
from honey.compiler import compile_model, ops
from honey.compiler.ops.common.epilogue import FuncEnum
from honey.frontend import Tensor
from honey.testing import detect_target
from honey.utils import shape_utils
from honey.testing.benchmark_honey import make_input_output_pools, run_benchmark

LOGGER = logging.getLogger(__name__)


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


class TestStridedLayerNormBenchmark(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_id = 0

    @unittest.skipIf(detect_target().in_ci_env(), "don't run benchmark in CI")
    def test_benchmark(self):
        for (
            input_nonbatch_shape,
            (start_indices, end_indices),
        ) in itertools.product(
            ((2048, 256), (2048, 512), (2048, 1024), (2048, 2048)),
            (((0, 0, 4), (None, None, 224)), ((0, 0, 3), (None, None, 223))),
        ):
            BATCH_SIZE = 2
            NUM_ITERS = 1000000
            NUM_WARMUP_ITERS = 10000
            INPUT_POOL_SIZE = 100
            _layernorm_common_params = {
                "input_nonbatch_shape": input_nonbatch_shape,
                "n_normalize_over_last_dims": 1,
                "gamma_is_none": None,
                "beta_is_none": None,
                "fuse_sigmoid_mul": False,
                "eps": 1e-5,
                "start_indices": start_indices,
                "end_indices": end_indices,
            }
            honey_module = build_honey_module(
                batch_sizes=(BATCH_SIZE,),
                workdir=uuid.uuid4().hex,
                test_id=self.test_id,
                **_layernorm_common_params,
            )
            self.test_id += 1
            inputs_pool, outputs_pool = make_input_output_pools(
                pool_size=INPUT_POOL_SIZE,
                eval_pt_func=lambda: eval_pt(
                    batch_size=BATCH_SIZE,
                    **_layernorm_common_params,
                ),
                input_filter_func=lambda k, v: not k.startswith("output")
                and v is not None,
                output_filter_func=lambda k, _: k.startswith("output"),
            )
            mean_runtime = run_benchmark(
                honey_module=honey_module,
                inputs_pool=inputs_pool,
                outputs_pool=outputs_pool,
                num_iters=NUM_ITERS,
                num_warmup_iters=NUM_WARMUP_ITERS,
            )
            benchmark_results = {
                "mean_runtime": mean_runtime,
                "input_nonbatch_shape": input_nonbatch_shape,
                "start_indices": start_indices,
                "end_indices": end_indices,
            }
            LOGGER.warning(
                f"Benchmark results {json.dumps(benchmark_results, separators=(',', ':'))}"
            )


if __name__ == "__main__":
    unittest.main()
