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
Codegen functions for perm021fc_crc_bias, which computes
[b, n, m](col) = bmm([1, k, n](col), [b, k, m](row)) + bias[n].
"""
from honey.backend import registry
from honey.backend.cuda.gemm_universal import (
    bmm_common,
    common,
    common_bias,
    perm021fc_crc,
)

# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703


def _get_problem_info(**kwargs):
    problem_args = {
        "beta_value": 1,
        "problem_dim_0": "N",
        "problem_dim_1": "M",
        "problem_dim_2": "K",
        "bias_ptr": "bias_ptr",
        "a_batch_stride": "0",
        "b_batch_stride": "K * M",
        "bias_batch_stride": "0",
        "c_batch_stride": "M * N",
        "lda": "N",
        "ldb": "M",
        "ldbias": "0",
        "ldc": "N",
    }
    for k, v in kwargs.items():
        problem_args[k] = v

    bmm_problem_info = bmm_common.Bmm_problem_info(**problem_args)
    return bmm_problem_info


@registry.reg("cuda.perm021fc_crc_bias.config")
def gemm_rcr_config(func_attrs, dtype="float16"):
    return perm021fc_crc.gemm_crc_config(func_attrs, dtype)


@registry.reg("cuda.perm021fc_crc_bias.gen_profiler")
def gen_profiler(func_attrs, workdir, profiler_filename, dim_info_dict):
    args_parser = bmm_common.ARGS_PARSER_TEMPLATE.render(
        a_dims=["1", "K", "N"], b_dims=["B", "K", "M"], c_dims=["B", "M", "N"]
    )

    problem_args = bmm_common.PROBLEM_ARGS_TEMPLATE.render(
        mm_info=_get_problem_info(
            alpha_value=func_attrs.get("alpha", 1),
        ),
    )

    return bmm_common.gen_profiler(
        func_attrs,
        workdir,
        profiler_filename,
        dim_info_dict,
        common_bias.SRC_TEMPLATE,
        problem_args,
        args_parser,
        bias_ptr_arg="memory_pool->RequestTensorByIdx(3)",
    )


@registry.reg("cuda.perm021fc_crc_bias.gen_function")
def gen_function(
    func_attrs,
    exec_cond_template,
    dim_info_dict,
):
    problem_args = bmm_common.PROBLEM_ARGS_TEMPLATE.render(
        mm_info=_get_problem_info(
            alpha_value=func_attrs.get("alpha", 1),
        ),
    )
    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    output_ndims = len(func_attrs["output_accessors"][0].original_shapes)

    return common.gen_function(
        func_attrs,
        common_bias.SRC_TEMPLATE,
        exec_cond_template,
        problem_args,
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        output_ndims=output_ndims,
        dim_info_dict=dim_info_dict,
    )


@registry.reg("cuda.perm021fc_crc_bias.func_decl")
def gen_function_decl(func_attrs):
    func_name = func_attrs["name"]
    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    return common_bias.FUNC_DECL_TEMPLATE.render(
        func_name=func_name, input_ndims=input_ndims, weight_ndims=weight_ndims
    )


@registry.reg("cuda.perm021fc_crc_bias.func_call")
def gen_function_call(func_attrs, indent="  "):
    bias = func_attrs["inputs"][2]
    return bmm_common.gen_function_call(
        func_attrs, indent, bias_ptr_arg=bias._attrs["name"]
    )


@registry.reg("cuda.perm021fc_crc_bias.filter")
def function_filter(cfg, func_attrs, ab_alignment):
    """Generates function filter.

    Parameters
    ----------
    cfg: str
        The filename generated for profiler.
    func_attrs : Dict
        Stores the operation attributes.
    ab_alignment:
        Input alignments.

    Returns
    -------
    bool
        If input cfg should be filtered.
    """
    return common.function_filter(cfg, func_attrs, ab_alignment)
