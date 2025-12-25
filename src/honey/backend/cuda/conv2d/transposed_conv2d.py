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
transposed conv2d op codegen
"""

from honey.backend import registry
from honey.backend.cuda.conv2d import common
from honey.backend.cuda.conv2d.conv2d import get_apply_special_config

# pylint: disable=C0103,C0415,W0613,C0301


@registry.reg("cuda.transposed_conv2d.config")
def transposed_conv2d_config(
    func_attrs,
    dtype="float16",
):
    is_bias = False
    is_bias_add = False
    is_transpose = True
    is_depthwise = False
    is_few_channels = False
    skip_simt_kernels = False
    if func_attrs["bias"] and not func_attrs["add"]:
        is_bias = True
    elif func_attrs["bias"] and func_attrs["add"]:
        is_bias_add = True
    if func_attrs["few_channels"]:
        skip_simt_kernels = True
        is_few_channels = True
    activation_op_name = None
    binary_op_name = None
    unary_op_name = None
    activation = func_attrs["activation"]
    if is_bias_add and activation is not None:
        if activation == "identity":
            activation_op_name = "Identity"
            binary_op_name = "Plus"
            unary_op_name = "Identity"
        elif activation == "relu":
            activation_op_name = "Identity"
            binary_op_name = "Plus"
            unary_op_name = "ReLu"
        elif activation == "hardswish":
            activation_op_name = "Identity"
            binary_op_name = "Add"
            unary_op_name = "HardSwish"
        else:
            raise NotImplementedError(f"is_bias_add with {activation=}.")
    f_apply_special_config = get_apply_special_config(
        few_channels=is_few_channels,
        transpose=is_transpose,
        depthwise=is_depthwise,
        activation_op_name=activation_op_name,
        binary_op_name=binary_op_name,
        unary_op_name=unary_op_name,
        dtype=dtype,
    )
    func_attrs["op_instance"] = common.extract_config(
        func_attrs=func_attrs,
        dtype=dtype,
        f_apply_special_config=f_apply_special_config,
        skip_simt_kernels=skip_simt_kernels,
        force_alignment=is_depthwise,
    )


@registry.reg("cuda.transposed_conv2d.gen_profiler")
def transposed_conv2d_gen_profiler(
    func_attrs,
    workdir,
    profiler_filename,
    shape_template,
):
    is_bias = False
    is_bias_add = False
    is_transpose = True
    is_depthwise = False
    if func_attrs["bias"] and not func_attrs["add"]:
        is_bias = True
    elif func_attrs["bias"] and func_attrs["add"]:
        is_bias_add = True
    return common.gen_profiler(
        func_attrs=func_attrs,
        workdir=workdir,
        profiler_filename=profiler_filename,
        shape_template=shape_template,
        is_bias=is_bias,
        is_transpose=is_transpose,
        instance_name_base="DeviceConvBwdInstance",
    )


@registry.reg("cuda.transposed_conv2d.gen_function")
def transposed_conv2d_gen_function(
    func_attrs,
    exec_cond_template,
    shape_eval_template,
    shape_save_template,
):
    is_bias = False
    is_bias_add = False
    is_transpose = True
    is_depthwise = False
    if func_attrs["bias"] and not func_attrs["add"]:
        is_bias = True
    elif func_attrs["bias"] and func_attrs["add"]:
        is_bias_add = True
    return common.gen_function(
        func_attrs=func_attrs,
        exec_cond_template=exec_cond_template,
        shape_eval_template=shape_eval_template,
        shape_save_template=shape_save_template,
        is_transpose=is_transpose,
        is_bias=is_bias,
    )


@registry.reg("cuda.transposed_conv2d.func_decl")
def transposed_conv2d_func_decl(
    func_attrs,
):
    is_bias = False
    is_bias_add = False
    is_transpose = True
    is_depthwise = False
    if func_attrs["bias"] and not func_attrs["add"]:
        is_bias = True
    elif func_attrs["bias"] and func_attrs["add"]:
        is_bias_add = True
    return common.gen_function_decl(
        func_attrs=func_attrs,
        is_bias=is_bias,
        is_bias_add=is_bias_add,
    )


@registry.reg("cuda.transposed_conv2d.func_call")
def transposed_conv2d_func_call(
    func_attrs,
    indent="  ",
):
    is_bias = False
    is_bias_add = False
    is_transpose = True
    is_depthwise = False
    if func_attrs["bias"] and not func_attrs["add"]:
        is_bias = True
    elif func_attrs["bias"] and func_attrs["add"]:
        is_bias_add = True
    return common.gen_function_call(
        func_attrs=func_attrs,
        indent=indent,
        is_transpose=is_transpose,
        is_bias=is_bias,
        is_bias_add=is_bias_add,
    )


@registry.reg("cuda.transposed_conv2d.filter")
def transposed_conv2d_filter(
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
