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
import unittest
from typing import List

import torch
from honey.compiler import compile_model, ops
from honey.frontend import IntImm, IntVar, Tensor
from honey.testing import detect_target
from honey.testing.test_utils import (
    get_random_torch_tensor,
    get_torch_empty_tensor,
)
from honey.utils import shape_utils, torch_utils


def build_honey_module(
    *,
    batch_sizes,
    input_nonbatch_shapes,
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
    test_name="strided_group_layernorm",
):
    target = detect_target()
    inputs = [
        Tensor(
            shape=[
                shape_utils.gen_int_var_min_max(values=batch_sizes, name="input_batch"),
                *shape,
            ],
            dtype=honey_dtype,
            name=f"input_{i}",
            is_input=True,
        )
        for i, shape in enumerate(input_nonbatch_shapes)
    ]
    sliced_inputs = [
        ops.dynamic_slice()(input_node, start_indices, end_indices)
        for input_node in inputs
    ]
    layernorm_weight_shapes = [
        n.shape()[-n_normalize_over_last_dims:] for n in sliced_inputs
    ]
    gammas = [
        (
            None
            if gamma_is_none
            else Tensor(shape=shape, dtype=honey_dtype, name=f"gamma_{i}", is_input=True)
        )
        for i, shape in enumerate(layernorm_weight_shapes)
    ]
    betas = [
        (
            None
            if beta_is_none
            else Tensor(shape=shape, dtype=honey_dtype, name=f"beta_{i}", is_input=True)
        )
        for i, shape in enumerate(layernorm_weight_shapes)
    ]
    layernorm_op = (
        ops.group_layernorm_sigmoid_mul() if fuse_sigmoid_mul else ops.group_layernorm()
    )
    outputs = layernorm_op(
        sliced_inputs,
        gammas=gammas,
        betas=betas,
        normalized_shapes=layernorm_weight_shapes,
        eps=eps,
    )
    for i, output in enumerate(outputs):
        output._attrs["is_output"] = True
        output._attrs["name"] = f"output_{i}"
    dll_name = f"test_{test_id}.so"
    return compile_model(
        outputs,
        target,
        workdir,
        test_name,
        dll_name=dll_name,
    )


def apply_pt_layernorm(
    *, input, normalized_shape, weight, bias, fuse_sigmoid_mul=False, eps=1e-5
):
    layernorm_output = torch.nn.functional.layer_norm(
        input=input,
        normalized_shape=normalized_shape,
        weight=weight,
        bias=bias,
        eps=eps,
    )
    if fuse_sigmoid_mul:
        output = torch.mul(input, torch.sigmoid(layernorm_output))
    else:
        output = layernorm_output
    return output


def eval_pt(
    *,
    batch_size,
    input_nonbatch_shapes,
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
    inputs = [
        torch.randn(batch_size, *shape, **dtype_device)
        for shape in input_nonbatch_shapes
    ]
    sliced_inputs = [
        x[[slice(i, j) for i, j in zip(start_indices, end_indices)]] for x in inputs
    ]
    layernorm_weight_shapes = [
        x.shape[-n_normalize_over_last_dims:] for x in sliced_inputs
    ]
    gammas = [
        None if gamma_is_none else torch.randn(shape, **dtype_device)
        for shape in layernorm_weight_shapes
    ]
    betas = [
        None if beta_is_none else torch.randn(shape, **dtype_device)
        for shape in layernorm_weight_shapes
    ]
    outputs = [
        apply_pt_layernorm(
            input=input,
            normalized_shape=normalized_shape,
            weight=weight,
            bias=bias,
            eps=eps,
            fuse_sigmoid_mul=fuse_sigmoid_mul,
        )
        for input, normalized_shape, weight, bias in zip(
            sliced_inputs, layernorm_weight_shapes, gammas, betas
        )
    ]
    return {
        **{f"input_{i}": x for i, x in enumerate(inputs)},
        **{f"gamma_{i}": x for i, x in enumerate(gammas)},
        **{f"beta_{i}": x for i, x in enumerate(betas)},
        **{f"output_{i}": x for i, x in enumerate(outputs)},
    }


class SliceGroupLayerNormTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(SliceGroupLayerNormTestCase, self).__init__(*args, **kwargs)
        self._test_id = 0

    def _test_slice_group_layer_norm(
        self,
        *,
        input_nonbatch_shapes: List[List[int]] = None,
        n_normalize_over_last_dims: int = 1,
        batch_sizes=(3, 4, 7, 11, 18),
        gamma_is_none=False,
        beta_is_none=False,
        fuse_sigmoid_mul=False,
        eps=1e-5,
        start_indices: List[int] = (0,),
        end_indices: List[int] = (None,),
        dtype: str = "float16",
    ):
        input_rank = 1 + len(input_nonbatch_shapes[0])
        if 1 == len(start_indices) and len(start_indices) != input_rank:
            start_indices = [start_indices[0]] * input_rank
        if 1 == len(end_indices) and len(end_indices) != input_rank:
            end_indices = [end_indices[0]] * input_rank

        _layernorm_common_params = {
            "input_nonbatch_shapes": input_nonbatch_shapes,
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
        )
        self._test_id += 1
        pt_dtype = torch_utils.string_to_torch_dtype(dtype)
        for batch_size in batch_sizes:
            pt_tensors = eval_pt(
                batch_size=batch_size, **_layernorm_common_params, dtype=pt_dtype
            )
            honey_inputs = {
                k: v
                for k, v in pt_tensors.items()
                if v is not None and not k.startswith("output")
            }
            honey_outputs = {
                k: torch.empty_like(v)
                for k, v in pt_tensors.items()
                if k.startswith("output")
            }
            honey_module.run_with_tensors(honey_inputs, honey_outputs)

            for k, v in honey_outputs.items():
                self.assertTrue(
                    torch.allclose(v, pt_tensors[k], atol=1e-2, rtol=1e-3),
                    f"max diff: {torch.max(v - pt_tensors[k]) if v.numel() > 0 else 0}, "
                    f"min diff: {torch.min(v - pt_tensors[k]) if v.numel() > 0 else 0}",
                )

    def _test_slice_group_layer_norm_kernels(
        self,
        **kwargs,
    ):
        for start_indices, end_indices, input_nonbatch_shapes in (
            # (cuda-half4) kernel
            ((0, 0, 0, 4), (None, None, None, 36), ((4, 1, 40), (4, 1, 40))),
            # (generic n < 1024) kernel
            ((0, 0, 0, 11), (None, None, None, 13), ((4, 1, 15), (4, 1, 15))),
            # (cuda-half; block size = 512) kernel
            ((0, 0, 0, 1), (None, None, None, 1026), ((4, 1, 1027), (4, 1, 1027))),
        ):
            self._test_slice_group_layer_norm(
                start_indices=start_indices,
                end_indices=end_indices,
                input_nonbatch_shapes=input_nonbatch_shapes,
                **kwargs,
            )

    def _test_middle_slice_group_layer_norm_kernels(
        self,
        **kwargs,
    ):
        for start_indices, end_indices, input_nonbatch_shapes in (
            # (cuda-half4) kernel
            ((0, 0, 4, 0), (None, None, 36, None), ((2, 40, 4), (2, 40, 4))),
            # (generic n < 1024) kernel
            ((0, 0, 11, 0), (None, None, 13, None), ((2, 15, 2), (2, 15, 2))),
            # (cuda-half; block size = 512) kernel
            ((0, 0, 1, 0), (None, None, 1026, None), ((2, 1027, 2), (2, 1027, 2))),
        ):
            self._test_slice_group_layer_norm(
                start_indices=start_indices,
                end_indices=end_indices,
                input_nonbatch_shapes=input_nonbatch_shapes,
                **kwargs,
            )

    def test_slice_group_layer_norm_float16(self):
        for (
            n_normalize_over_last_dims,
            gamma_is_none,
            beta_is_none,
        ) in itertools.product(
            (1, 3),
            (True, False),
            (True, False),
        ):
            self._test_slice_group_layer_norm_kernels(
                n_normalize_over_last_dims=n_normalize_over_last_dims,
                gamma_is_none=gamma_is_none,
                beta_is_none=beta_is_none,
                fuse_sigmoid_mul=False,
            )

    def test_middle_slice_group_layer_norm_float16(self):
        for (
            n_normalize_over_last_dims,
            gamma_is_none,
            beta_is_none,
        ) in itertools.product(
            (2, 3),
            (True, False),
            (True, False),
        ):
            self._test_middle_slice_group_layer_norm_kernels(
                n_normalize_over_last_dims=n_normalize_over_last_dims,
                gamma_is_none=gamma_is_none,
                beta_is_none=beta_is_none,
                fuse_sigmoid_mul=False,
            )

    def test_slice_group_layer_norm_fuse_sigmoid_mul_float16(self):
        for (
            n_normalize_over_last_dims,
            gamma_is_none,
            beta_is_none,
        ) in itertools.product(
            (1, 3),
            (True, False),
            (True, False),
        ):
            self._test_slice_group_layer_norm_kernels(
                n_normalize_over_last_dims=n_normalize_over_last_dims,
                gamma_is_none=gamma_is_none,
                beta_is_none=beta_is_none,
                fuse_sigmoid_mul=True,
            )

    def test_middle_slice_group_layer_norm_fuse_sigmoid_mul_float16(self):
        for (
            n_normalize_over_last_dims,
            gamma_is_none,
            beta_is_none,
        ) in itertools.product(
            (2, 3),
            (True, False),
            (True, False),
        ):
            self._test_middle_slice_group_layer_norm_kernels(
                n_normalize_over_last_dims=n_normalize_over_last_dims,
                gamma_is_none=gamma_is_none,
                beta_is_none=beta_is_none,
                fuse_sigmoid_mul=True,
            )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_slice_group_layer_norm_float(self):
        self._test_slice_group_layer_norm_kernels(
            n_normalize_over_last_dims=3,
            gamma_is_none=True,
            beta_is_none=True,
            fuse_sigmoid_mul=False,
            dtype="float32",
        )
        self._test_middle_slice_group_layer_norm_kernels(
            n_normalize_over_last_dims=2,
            gamma_is_none=True,
            beta_is_none=False,
            fuse_sigmoid_mul=False,
            dtype="float32",
        )
        self._test_slice_group_layer_norm_kernels(
            n_normalize_over_last_dims=1,
            gamma_is_none=False,
            beta_is_none=True,
            fuse_sigmoid_mul=True,
            dtype="float32",
        )
        self._test_middle_slice_group_layer_norm_kernels(
            n_normalize_over_last_dims=3,
            gamma_is_none=False,
            beta_is_none=False,
            fuse_sigmoid_mul=True,
            dtype="float32",
        )

    @unittest.skipIf(detect_target().name() == "rocm", "Not supported by ROCM.")
    def test_group_layernorm_no_cuda_illegal_memory_access(self):
        """
        This subgraph has led to CUDA illegal memory issues before.
        Adding it as a unit test to ensure there are no regressions.
        """
        batch_size = IntVar(values=[1, 2048], name="batch_size")

        unsqueeze_46_0 = Tensor(
            shape=[batch_size, 7680, 1],
            is_input=True,
            name="unsqueeze_46_0",
        )
        unsqueeze_58_0 = Tensor(
            shape=[batch_size, 7680, 1],
            is_input=True,
            name="unsqueeze_58_0",
        )
        unsqueeze_70_0 = Tensor(
            shape=[batch_size, 7680, 1],
            is_input=True,
            name="unsqueeze_70_0",
        )
        unsqueeze_131_0 = Tensor(
            shape=[batch_size, 3, 1],
            is_input=True,
            name="unsqueeze_131_0",
        )
        main_module_base_forward_module_over_arch_bottom_arch_list_1_dime_shared_arch_layer_norm__norm_weight = Tensor(
            shape=[IntImm(256)],
            is_input=True,
            name="main_module_base_forward_module_over_arch_bottom_arch_list_1_dime_shared_arch_layer_norm__norm_weight",
        )
        main_module_base_forward_module_over_arch_bottom_arch_list_1_dime_shared_arch_layer_norm__norm_bias = Tensor(
            shape=[IntImm(256)],
            is_input=True,
            name="main_module_base_forward_module_over_arch_bottom_arch_list_1_dime_shared_arch_layer_norm__norm_bias",
        )

        unsqueeze_83_0 = Tensor(
            shape=[batch_size, 7680, 1],
            is_input=True,
            name="unsqueeze_83_0",
        )
        unsqueeze_95_0 = Tensor(
            shape=[batch_size, 7680, 1],
            is_input=True,
            name="unsqueeze_95_0",
        )
        unsqueeze_107_0 = Tensor(
            shape=[batch_size, 7680, 1],
            is_input=True,
            name="unsqueeze_107_0",
        )
        unsqueeze_358_0 = Tensor(
            shape=[batch_size, 3, 1],
            is_input=True,
            name="unsqueeze_358_0",
        )
        main_module_base_forward_module_over_arch_bottom_arch_list_0_dime_shared_arch_layer_norm__norm_weight = Tensor(
            shape=[IntImm(256)],
            is_input=True,
            name="main_module_base_forward_module_over_arch_bottom_arch_list_0_dime_shared_arch_layer_norm__norm_weight",
        )
        main_module_base_forward_module_over_arch_bottom_arch_list_0_dime_shared_arch_layer_norm__norm_bias = Tensor(
            shape=[IntImm(256)],
            is_input=True,
            name="main_module_base_forward_module_over_arch_bottom_arch_list_0_dime_shared_arch_layer_norm__norm_bias",
        )

        concatenate_71_0 = ops.concatenate()(
            inputs=[unsqueeze_46_0, unsqueeze_58_0, unsqueeze_70_0],
            dim=2,
        )
        bmm_rrr_132_0 = ops.bmm_rrr()(concatenate_71_0, unsqueeze_131_0)
        reshape_133_0 = ops.reshape()(bmm_rrr_132_0, shape=[-1, 30, 256])
        layernorm_134_0 = ops.layernorm(normalized_shape=[IntImm(256)])(
            reshape_133_0,
            main_module_base_forward_module_over_arch_bottom_arch_list_1_dime_shared_arch_layer_norm__norm_weight,
            main_module_base_forward_module_over_arch_bottom_arch_list_1_dime_shared_arch_layer_norm__norm_bias,
        )
        permute021_136_0 = ops.permute021()(layernorm_134_0)

        concatenate_108_0 = ops.concatenate()(
            inputs=[unsqueeze_83_0, unsqueeze_95_0, unsqueeze_107_0],
            dim=2,
        )
        bmm_rrr_359_0 = ops.bmm_rrr()(concatenate_108_0, unsqueeze_358_0)
        reshape_360_0 = ops.reshape()(bmm_rrr_359_0, shape=[-1, 30, 256])
        layernorm_361_0 = ops.layernorm(normalized_shape=[IntImm(256)])(
            reshape_360_0,
            main_module_base_forward_module_over_arch_bottom_arch_list_0_dime_shared_arch_layer_norm__norm_weight,
            main_module_base_forward_module_over_arch_bottom_arch_list_0_dime_shared_arch_layer_norm__norm_bias,
        )
        permute021_363_0 = ops.permute021()(layernorm_361_0)

        outputs = [permute021_136_0, permute021_363_0]

        for i, output in enumerate(outputs):
            output._attrs["is_output"] = True
            output._attrs["name"] = f"output_{i}"

        model = compile_model(
            outputs,
            detect_target(),
            "./tmp",
            "test_group_layernorm_repro",
        )

        pt_inputs = {
            "unsqueeze_46_0": get_random_torch_tensor(shape=[1024, 7680, 1]),
            "unsqueeze_58_0": get_random_torch_tensor(shape=[1024, 7680, 1]),
            "unsqueeze_70_0": get_random_torch_tensor(shape=[1024, 7680, 1]),
            "unsqueeze_131_0": get_random_torch_tensor(shape=[1024, 3, 1]),
            "main_module_base_forward_module_over_arch_bottom_arch_list_1_dime_shared_arch_layer_norm__norm_weight": get_random_torch_tensor(
                shape=[256]
            ),
            "main_module_base_forward_module_over_arch_bottom_arch_list_1_dime_shared_arch_layer_norm__norm_bias": get_random_torch_tensor(
                shape=[256]
            ),
            "unsqueeze_83_0": get_random_torch_tensor(shape=[1024, 7680, 1]),
            "unsqueeze_95_0": get_random_torch_tensor(shape=[1024, 7680, 1]),
            "unsqueeze_107_0": get_random_torch_tensor(shape=[1024, 7680, 1]),
            "unsqueeze_358_0": get_random_torch_tensor(shape=[1024, 3, 1]),
            "main_module_base_forward_module_over_arch_bottom_arch_list_0_dime_shared_arch_layer_norm__norm_weight": get_random_torch_tensor(
                shape=[256]
            ),
            "main_module_base_forward_module_over_arch_bottom_arch_list_0_dime_shared_arch_layer_norm__norm_bias": get_random_torch_tensor(
                shape=[256]
            ),
        }
        pt_outputs = {
            "output_0": get_torch_empty_tensor(shape=[1024, 256, 30]),
            "output_1": get_torch_empty_tensor(shape=[1024, 256, 30]),
        }

        model.run_with_tensors(pt_inputs, pt_outputs)


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
