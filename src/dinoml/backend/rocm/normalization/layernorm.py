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
Layernorm codegen for ROCM.
"""

from collections import OrderedDict
from hashlib import sha1
import os
from typing import Any, Dict

import jinja2

from dinoml.backend import registry
from dinoml.backend.backend_spec import ROCMSpec
from dinoml.backend.rocm.normalization import norm_common
from dinoml.backend.target import Target

from dinoml.compiler.base import IntImm

EXTRA_HEADERS = jinja2.Template(
    """
#include "ck/tensor_operation/gpu/device/impl/device_normalization_fwd_impl.hpp"
"""
)

FUNC_CALL_FP16_PARAM_TEMPLATE = jinja2.Template(
    "reinterpret_cast<{{dtype}}*>(({{name}}))"
)

TENSOR_DECL_TEMPLATE = jinja2.Template(
    """
  int64_t ptr_sz = in_{{ range(rank)|join(' * in_') }};

  int64_t norm_dim = in_{{rank - 1}};

  // TODO: special pool size for 8M L2 cache
  // need to tune it for other devices
  int64_t mem_pool_sz = std::max(2,  std::min(64, int((1 << 23) / ptr_sz)));

  memory_pool->AllocateHalfTensor(ptr_sz, mem_pool_sz);  // in: index 0
  memory_pool->AllocateHalfTensor(ptr_sz, mem_pool_sz);  // out: index 1
  memory_pool->AllocateHalfTensor(norm_dim, mem_pool_sz);  // gamma: index 2
  memory_pool->AllocateHalfTensor(norm_dim, mem_pool_sz);  // beta: index 3

"""
)

SHAPE_EVAL_TEMPLATE = jinja2.Template(
    """
    int M = *in_{{ range(rank - 1)|join(' * *in_') }};
    int N = *in_{{rank - 1}};
    """
)

EXEC_TEMPLATE = jinja2.Template(
    """
    std::vector<ck::index_t> i_inStrides;
    std::vector<ck::index_t> i_outStrides;
    {% if input_strides is defined %}
    i_inStrides.push_back({{input_strides[-2]}});
    i_inStrides.push_back({{input_strides[-1]}});
    {% else %}
    i_inStrides.push_back(N);
    i_inStrides.push_back(1);
    {% endif %}

    {% if output_strides is defined %}
    i_outStrides.push_back({{output_strides[-2]}});
    i_outStrides.push_back({{output_strides[-1]}});
    {% else %}
    i_outStrides.push_back(N);
    i_outStrides.push_back(1);
    {% endif %}

    auto device_instance = {{instance}}{};
    auto argument_ptr = device_instance.MakeArgumentPointer(
        {M, N},
        i_inStrides,
        std::vector<ck::index_t>{0, 1},
        std::vector<ck::index_t>{0, 1},
        i_outStrides,
        std::vector<ck::index_t>{0},
        std::vector<ck::index_t>{0},
        {1},
        {{eps}},
        static_cast<{{dtype}} *>(input) + {{ input_offset if input_offset is defined else 0 }},
        static_cast<{{dtype}} *>(gamma),
        static_cast<{{dtype}} *>(beta),
        static_cast<{{dtype}} *>(output) + {{ output_offset if output_offset is defined else 0 }},
        nullptr,
        nullptr,
        ck::tensor_operation::element_wise::PassThrough{}
    );

    if(!device_instance.IsSupportedArgument(argument_ptr.get()))
    {
        LOG(FATAL) << "wrong! " << device_instance.GetTypeString() << " with the specified compilation parameters does not support this Layernorm problem.";
    };
    auto invoker_ptr = device_instance.MakeInvokerPointer();
    invoker_ptr->Run(argument_ptr.get(), StreamConfig{stream, false});
    return;
"""
)
FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}({{dtype}}* input,
                   {{dtype}}* gamma,
                   {{dtype}}* beta,
                   {{dtype}}* output,
{% for idx in range(input_ndim) %}
                   int64_t* in_{{idx}},
{% endfor %}
                   hipStream_t stream)
    """
)


FUNC_DECL = jinja2.Template(
    """
    {{func_signature}};
    """
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}   {{input}},
{{indent}}   {{gamma}},
{{indent}}   {{beta}},
{{indent}}   {{output}},
{% for name in input_dim_names %}
{{indent}}    const_cast<int64_t *>(&{{name}}),
{% endfor %}
{{indent}}   stream
{{indent}});
    """
)


@registry.reg("rocm.layernorm.config")
def extract_config(func_attrs):
    """Extract (operation name, operation instance) pair
    from all operation candidates.

    Parameters
    ----------
    op_kind : ck_lib.library.OperationKind
        Operation kind.
    extra_kind : ck_lib.library.[AnyKind]
        Used to as extra flag to distinguish kernels.
        E.g. bias_add_relu vs. add_relu_bias
    f_prop_op: function
        Used to filter operation.

    Returns
    -------
    Dict
        Extracted (operation name, operation instance) pair.
    """
    import dinoml.utils.ck_lib as ck_lib

    x = func_attrs["inputs"][0]
    spec = ROCMSpec()
    lib_dtype = spec.dtype_to_lib_type(x._attrs["dtype"])

    if lib_dtype == "float":
        data_type = ck_lib.library.DataType.f32
        acc_type = ck_lib.library.DataType.f32
    elif lib_dtype == "ck::half_t":
        data_type = ck_lib.library.DataType.f16
        acc_type = ck_lib.library.DataType.f32
        # check target use fp16 acc
        if (
            "use_fp16_acc" in Target.current()._kwargs
            and Target.current().name() != "rocm"
        ):
            if Target.current()._kwargs["use_fp16_acc"]:
                acc_type = ck_lib.library.DataType.f16
    elif lib_dtype == "ck::bhalf_t":
        data_type = ck_lib.library.DataType.bf16
        acc_type = ck_lib.library.DataType.f32
    else:
        raise RuntimeError(f"Unsupported dtype {lib_dtype}")

    def f_proc_op(op: ck_lib.layernorm_operation.LayerNormOperation):
        if op.In == data_type and op.Out == data_type and op.acc_dtype == acc_type:
            return op
        return None

    op_kind = ck_lib.library.OperationKind.LayerNorm
    extra_kind = 2
    extract_ops = list(Target.current()._operators[op_kind][extra_kind].items())
    layernorm_ops = OrderedDict()
    for key, value in extract_ops:
        op = value[0]
        if f_proc_op(op) is not None:
            layernorm_ops[key] = op
    func_attrs["op_instance"] = layernorm_ops


def get_func_signature_profiler(func_attrs: Dict[str, Any]) -> str:
    return FUNC_SIGNATURE.render(
        func_name=func_attrs["name"],
        dtype="void",
        input_ndim=2,
    ).strip()


PROFILER_TEMPLATE = jinja2.Template(
    """
#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <random>
#include <rocrand/rocrand.h>
#include "logging.h"
//include "ck/utility/print.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/utility/reduction_operator.hpp"
{{extra_headers}}

{{extra_code}}

size_t GLOBAL_WORKSPACE_SIZE = 0;

{{structs_def}}

{{all_instances}}

template <typename Instance>
int benchmark_norm(
    Instance& device_instance,
    const char* op_name,
    ProfilerMemoryPool* memory_pool,
    int64_t in_0,
    int64_t in_1,
    float eps,
    hipStream_t stream)
{
    // in_0 = M, in_1 = N (reduction dim)
    int M = static_cast<int>(in_0);
    int N = static_cast<int>(in_1);

    std::vector<ck::index_t> i_inStrides = {static_cast<ck::index_t>(N), 1};
    std::vector<ck::index_t> i_outStrides = {static_cast<ck::index_t>(N), 1};

    auto argument_ptr = device_instance.MakeArgumentPointer(
        {M, N},
        i_inStrides,
        std::vector<ck::index_t>{0, 1},
        std::vector<ck::index_t>{0, 1},
        i_outStrides,
        std::vector<ck::index_t>{0},
        std::vector<ck::index_t>{0},
        {1},
        eps,
        memory_pool->RequestHalfTensorByIdx(0), // input
        memory_pool->RequestHalfTensorByIdx(2), // gamma
        memory_pool->RequestHalfTensorByIdx(3), // beta
        memory_pool->RequestHalfTensorByIdx(1), // output
        nullptr,
        nullptr,
        ck::tensor_operation::element_wise::PassThrough{}
    );

    if(!device_instance.IsSupportedArgument(argument_ptr.get()))
    {
        return -1;
    }

    auto invoker = device_instance.MakeInvokerPointer();

    // warmup
    for(int i = 0; i < 3; ++i)
    {
        invoker->Run(argument_ptr.get(), StreamConfig{stream, false});
    }

    KernelTimerImpl timer;
    timer.Start();
    for(int i = 0; i < 5; ++i)
    {
        invoker->Run(argument_ptr.get(), StreamConfig{stream, false});
    }
    timer.End();

    std::cout << "OP:" << op_name
              << ",TIME:" << timer.GetElapsedTime()
              << ",WS:" << GLOBAL_WORKSPACE_SIZE
              << std::endl
              << std::flush;

    return 0;
}

int main(int argc, char** argv) {
  {{args_parse}}
  auto memory_pool = std::make_unique<ProfilerMemoryPool>();
  hipStream_t stream = nullptr;
  {{tensor_decl}}
  {% for op_name, inst_name in instances %}
{
    {{inst_name}} device_instance;
    benchmark_norm(
        device_instance,
        "{{op_name}}",
        memory_pool.get(),
        in_0, in_1,
        {{eps}},
        stream);
}
{% endfor %}
}
"""
)


def gen_profiler(
    func_attrs: Dict[str, Any],
    workdir: str,
    rank: int,
    extra_header_template: jinja2.Template,
    extra_code: str = "",
    profile_filename=None,
    indent: str = "  ",
) -> str:
    """Generates standalone executables for profiler.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.
    workdir : str
        Directory to store the generated outputs.
    rank: int
        Rank of the input tensor. If using [M, N] in exec_key, the rank here
        must be 2 because if implies that the inputs are reshaped for profiling.
        For code gen, the real shapes are used.
    exec_template : jinja2.Template
        Execution block template.
    tensor_decl_template: jinja2.Template
        Tensor declaration template.
    extra_header_template : jinja2.Template
        Extra header template.
    indent : str, optional
        Indent for codegen, target dependent e.g. C++, python, etc., by default "  ".
    """
    x = func_attrs["inputs"][0]
    spec = ROCMSpec()
    lib_dtype = spec.dtype_to_lib_type(x._attrs["dtype"])

    op_type = func_attrs["op"]
    op_instance = func_attrs["op_instance"]
    eps = func_attrs.get("eps", "1e-5")

    all_instances_decl = ""
    instance_names = []

    for idx, (op_name, op) in enumerate(op_instance.items()):
        config = norm_common.emit_instance(op)
        config_name = norm_common.extract_config_name(config)
        inst_name = f"DeviceInstance_{idx}"

        all_instances_decl += norm_common.INSTANCE_TEMPLATE.render(
            name=inst_name,
            config_name=config_name,
            config=config,
        )
        instance_names.append((op_name, inst_name))

    structs_def = norm_common.STRUCTS_DEF_TEMPLATE.render(dtype=lib_dtype)
    args_parse = norm_common.ARGS_PARSE_TEMPLATE.render(rank=rank)
    tensor_decl = TENSOR_DECL_TEMPLATE.render(rank=rank)

    file_pairs = []
    code = PROFILER_TEMPLATE.render(
        structs_def=structs_def,
        args_parse=args_parse,
        tensor_decl=tensor_decl,
        instances=instance_names,
        all_instances=all_instances_decl,
        extra_headers=extra_header_template.render(),
        extra_code=extra_code,
        eps=eps,
    )

    prefix = os.path.join(workdir, "profiler", op_type)
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    src_path = os.path.join(prefix, profile_filename + ".cpp")
    obj_path = os.path.join(prefix, profile_filename)
    if os.path.exists(obj_path):
        return
    with open(src_path, "w") as fo:
        fo.write(code)
    file_pairs.append((src_path, obj_path))
    return file_pairs


@registry.reg("rocm.layernorm.gen_profiler")
def layernorm_gen_profiler(
    func_attrs: Dict[str, Any],
    workdir: str,
    indent: str = "  ",
    profile_filename=None,
) -> str:
    """Generates standalone executables for profiler.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.
    workdir : str
        Directory to store the generated outputs.
    indent : str, optional
        Indent for codegen, target dependent e.g. C++, python, etc., by default "  ".
    """
    dim = -1
    shapes = func_attrs["inputs"][0]._attrs["shape"]

    assert isinstance(shapes[dim], IntImm), (
        "layernorm requires reduction dim to be static"
    )
    return gen_profiler(
        func_attrs,
        workdir,
        rank=2,
        extra_header_template=EXTRA_HEADERS,
        extra_code="",
        profile_filename=profile_filename,
        indent=indent,
    )


# This function has diverged from norm_common.gen_function
# due to the change to the profiler exec_key
# TODO: merge with norm_common.gen_function after fixing softmax
def gen_function(
    func_attrs: Dict[str, Any],
    shape_eval_template: jinja2.Template,
    exec_template: jinja2.Template,
    extra_header_template: jinja2.Template,
    get_func_signature: Any,
) -> str:
    """Generate function body.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.
    exec_template : jinja2.Template
        Execution block template.
    extra_header_template : jinja2.Template
        Extra header template.

    Returns
    -------
    str
        The rendered template of generated function body.
    """
    x = func_attrs["inputs"][0]
    spec = ROCMSpec()
    lib_dtype = spec.dtype_to_lib_type(x._attrs["dtype"])

    rank = func_attrs["inputs"][0]._rank()
    eps = func_attrs.get("eps", "1e-5")

    exec_path = func_attrs["exec_path"]
    op_instance = func_attrs["op_instance"]

    inst_def_flag = set()
    instances = {}
    instance_decl = ""
    for exec_item in exec_path.values():
        fname = "f" + sha1(exec_item.exec_cond.encode()).hexdigest()
        algo = exec_item.algo
        if algo not in inst_def_flag:
            config = norm_common.emit_instance(op_instance[algo])
            inst_def_flag.add(algo)
        else:
            config = ""
        inst = norm_common.INSTANCE_TEMPLATE.render(
            config=config,
            name=fname,
            config_name=norm_common.extract_config_name(config),
        )
        instances[exec_item.exec_cond] = inst
        instance_decl += inst

    shape_eval = shape_eval_template.render(rank=rank) if shape_eval_template else ""
    exec_cond_template = func_attrs["exec_cond_template"]
    exec_paths = ""
    for key, _ in instances.items():
        fname = "f" + sha1(key.encode()).hexdigest()
        program = exec_template.render(
            instance=fname,
            dtype=lib_dtype,
            reduce_dims=rank - 1,
            eps=eps,
        )
        exec_inst = exec_cond_template.render(indent="  ", cond=key, program=program)
        exec_paths += exec_inst

    return norm_common.FUNC_TEMPLATE.render(
        instances_decl=instance_decl,
        func_signature=get_func_signature(func_attrs),
        shape_eval=shape_eval,
        exec_paths=exec_paths,
        extra_headers=extra_header_template.render(),
    )


@registry.reg("rocm.layernorm.gen_function")
def layernorm_gen_function(func_attrs: Dict[str, Any]) -> str:
    """Generate function body.

    Parameters
    ----------
    func_attrs : Dict
        Operation attributes.

    Returns
    -------
    str
        The rendered template of generated function body.
    """
    dim = -1
    shapes = func_attrs["inputs"][0]._attrs["shape"]

    assert isinstance(shapes[dim], IntImm), (
        "layernorm requires reduction dim to be static"
    )

    return gen_function(
        func_attrs,
        SHAPE_EVAL_TEMPLATE,
        EXEC_TEMPLATE,
        EXTRA_HEADERS,
        get_func_signature,
    )


def get_func_signature(func_attrs: Dict[str, Any]) -> str:
    input_ndim = func_attrs["inputs"][0]._rank()
    return FUNC_SIGNATURE.render(
        func_name=func_attrs["name"],
        dtype="void",
        input_ndim=input_ndim,
    ).strip()


@registry.reg("rocm.layernorm.func_decl")
def layernorm_gen_function_decl(func_attrs: Dict[str, Any]):
    return FUNC_DECL.render(func_signature=get_func_signature(func_attrs))


@registry.reg("rocm.layernorm.func_call")
def layernorm_gen_function_call(func_attrs, indent="  "):
    assert len(func_attrs["outputs"]) == 1
    assert len(func_attrs["inputs"]) == 3

    x = func_attrs["inputs"][0]
    spec = ROCMSpec()
    lib_dtype = spec.dtype_to_lib_type(x._attrs["dtype"])

    input_name = FUNC_CALL_FP16_PARAM_TEMPLATE.render(
        name=func_attrs["inputs"][0]._attrs["name"], dtype=lib_dtype
    )
    gamma_name = FUNC_CALL_FP16_PARAM_TEMPLATE.render(
        name=func_attrs["inputs"][1]._attrs["name"], dtype=lib_dtype
    )
    beta_name = FUNC_CALL_FP16_PARAM_TEMPLATE.render(
        name=func_attrs["inputs"][2]._attrs["name"], dtype=lib_dtype
    )
    output_name = FUNC_CALL_FP16_PARAM_TEMPLATE.render(
        name=func_attrs["outputs"][0]._attrs["name"], dtype=lib_dtype
    )

    shapes = func_attrs["inputs"][0]._attrs["shape"]
    assert len(shapes) >= 2, (
        f"LayerNorm only supports input with rank >= 2, current rank: {len(shapes)}"
    )

    input_dim_names = [shape._attrs["name"] for shape in shapes]

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        input=input_name,
        gamma=gamma_name,
        beta=beta_name,
        output=output_name,
        input_dim_names=input_dim_names,
        indent=indent,
    )
