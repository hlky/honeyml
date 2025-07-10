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
Codegen for conv2d.
"""
from honey.backend import registry
from honey.backend.cuda.conv2d import common
from honey.utils import alignment

# pylint: disable=C0103,C0415,W0613,C0301

BIAS_ACT_EXTRA_HEADER = """
#include <cutlass/epilogue/thread/linear_combination_bias_relu.h>
#include <cutlass/epilogue/thread/linear_combination_hardswish.h>
#include <cutlass/epilogue/thread/linear_combination_silu.h>
"""

BIAS_ADD_ACT_EXTRA_HEADER = """
#include <cutlass/conv/kernel/default_conv2d_fprop_with_broadcast.h>
#include <cutlass/epilogue/thread/linear_combination_residual_block.h>
"""

def get_apply_special_config(few_channels: bool = False, transpose: bool = False, depthwise: bool = False, dtype="float16"):
    def apply_special_config(func_attrs, op):
        import honey.utils.cutlass_lib as cutlass_lib

        if depthwise:
            op.iterator_algorithm = cutlass_lib.library.IteratorAlgorithm.Analytic
            op.A.alignment = 1
            op.B.alignment = 1
            op.tile_description.stages = 2
            op.tile_description.math_instruction.instruction_shape = [1, 1, 1]
            op.tile_description.threadblock_shape[-1] = 8

        if transpose:
            op.group_mode = cutlass_lib.library.GroupMode.NoneGroup 

        if few_channels:
            x = func_attrs["inputs"][0]
            in_ch = x._attrs["shape"][-1]._attrs["values"][0]

            # Make sure to use NoneGroup here. Otherwise, we'll generate Conv2dGroupFprop,
            # which doesn't have template specializations for either of the iterator
            # algorithms below, resulting in "incomplete type is not allowed" errors.
            op.group_mode = cutlass_lib.library.GroupMode.NoneGroup
            if in_ch == 3:
                # By default we don't use it since the perf is worse than pad4+fixchannel
                op.iterator_algorithm = cutlass_lib.library.IteratorAlgorithm.FewChannels
                op.A.alignment = 1
                op.B.alignment = 1
                op.tile_description.stages = 2
            elif in_ch in alignment.get_alignments(dtype):
                op.iterator_algorithm = cutlass_lib.library.IteratorAlgorithm.FixedChannels
                op.A.alignment = in_ch
                op.B.alignment = in_ch
                op.tile_description.stages = 3
        return op
    return apply_special_config

@registry.reg("cuda.conv2d.config")
def conv2d_config(
    func_attrs,
    dtype="float16",
    is_bias=False,
    is_bias_add=False,
    is_transpose=False,
    is_depthwise=False,
    is_few_channels=False,
    skip_simt_kernels=False,
    f_apply_special_config=None,
    op_kind=None,
    op_layout=None,
):
    """Populates conv2d cutlass configs into 'op_instance' field."""
    if is_few_channels:
        skip_simt_kernels = True
    f_apply_special_config = get_apply_special_config(few_channels=is_few_channels, transpose=is_transpose, depthwise=is_depthwise, dtype=dtype)
    func_attrs["op_instance"] = common.extract_config(
        func_attrs=func_attrs,
        dtype=dtype,
        f_apply_special_config=f_apply_special_config,
        skip_simt_kernels=skip_simt_kernels,
    )


@registry.reg("cuda.conv2d.gen_profiler")
def conv2d_gen_profiler(
    func_attrs,
    workdir,
    profiler_filename,
    shape_template,
    is_bias=False,
    is_bias_add=False,
    is_transpose=False,
    is_depthwise=False,
    extra_header="",
):
    """Codegen for conv2d profiler."""
    if is_bias:
        extra_header = BIAS_ACT_EXTRA_HEADER
    if is_bias_add:
        extra_header = BIAS_ADD_ACT_EXTRA_HEADER
    return common.gen_profiler(
        func_attrs=func_attrs,
        workdir=workdir,
        profiler_filename=profiler_filename,
        shape_template=shape_template,
        is_bias=is_bias,
        is_bias_add=is_bias_add,
        is_transpose=is_transpose,
        is_depthwise=is_depthwise,
        extra_header=extra_header,
    )


@registry.reg("cuda.conv2d.gen_function")
def conv2d_gen_function(
    func_attrs,
    exec_cond_template,
    shape_eval_template,
    shape_save_template,
    is_bias=False,
    is_bias_add=False,
    is_transpose=False,
    is_depthwise=False,
    extra_header="",
):
    """Codegen for conv2d function."""
    if is_bias:
        extra_header = BIAS_ACT_EXTRA_HEADER
    if is_bias_add:
        extra_header = BIAS_ADD_ACT_EXTRA_HEADER
    return common.gen_function(
        func_attrs=func_attrs,
        exec_cond_template=exec_cond_template,
        shape_eval_template=shape_eval_template,
        shape_save_template=shape_save_template,
        is_bias=is_bias,
        is_bias_add=is_bias_add,
        is_transpose=is_transpose,
        is_depthwise=is_depthwise,
        extra_header=extra_header,
    )


@registry.reg("cuda.conv2d.func_decl")
def conv2d_func_decl(
    func_attrs,
    is_bias=False,
    is_bias_add=False,
):
    """Codegen for conv2d function declaration."""
    return common.gen_function_decl(
        func_attrs=func_attrs,
        is_bias=is_bias,
        is_bias_add=is_bias_add,
    )


@registry.reg("cuda.conv2d.func_call")
def conv2d_func_call(
    func_attrs,
    indent="  ",
    is_bias=False,
    is_bias_add=False,
    is_transpose=False,
):
    """Codegen for conv2d function call."""
    return common.gen_function_call(
        func_attrs=func_attrs,
        indent=indent,
        is_bias=is_bias,
        is_bias_add=is_bias_add,
        is_transpose=is_transpose,
    )


@registry.reg("cuda.conv2d.filter")
def conv2d_filter(
    cfg,
    func_attrs,
    x_shape,
):
    """Generates function filter.

    Parameters
    ----------
    cfg: str
        The filename generated for profiler.
    func_attrs : Dict
        Stores the operation attributes.
    x_shape:
        Input shapes.

    Returns
    -------
    bool
        If input cfg should be filtered.
    """
    return common.function_filter(
        cfg=cfg,
        func_attrs=func_attrs,
        x_shape=x_shape,
    )
