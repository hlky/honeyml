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
Groupnorm codegen for ROCM.
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
from dinoml.utils.shape_utils import get_shape

EXTRA_HEADERS = jinja2.Template(
    """
#include "ck/tensor_operation/gpu/device/impl/device_normalization_fwd_impl.hpp"
"""
)

EXTRA_CODE_TEMPLATE = jinja2.Template(
    """
{%if use_swish %}
struct YElementOp
{
    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const
    {
        static_assert(ck::is_same<X, float>::value || ck::is_same<X, double>::value ||
                          ck::is_same<X, {{dtype}}>::value,
                      "Data type is not supported by this operation!");

        static_assert(ck::is_same<Y, float>::value || ck::is_same<Y, double>::value ||
                          ck::is_same<Y, {{dtype}}>::value,
                      "Data type is not supported by this operation!");

        X a;

        ck::tensor_operation::element_wise::Sigmoid{}(a, x);

        y = ck::type_convert<Y>(x * a);
    };
};


{% else %}

using YElementOp   = ck::tensor_operation::element_wise::PassThrough;

{% endif %}
"""
)

FUNC_CALL_FP16_PARAM_TEMPLATE = jinja2.Template(
    "reinterpret_cast<{{dtype}}*>(({{name}}))"
)

TENSOR_DECL_TEMPLATE = jinja2.Template(
    """
  // N, H, W, G, C
  const int64_t N = in_0;
  const int64_t H = in_1;
  const int64_t W = in_2;
  const int64_t G = in_3;
  const int64_t C = in_4 ;
  int64_t ptr_sz = N * H * W * G * C;

  // TODO: special pool size for 8M L2 cache
  // need to tune it for other devices
  int64_t mem_pool_sz = std::max(2,  std::min(64, int((1 << 23) / ptr_sz)));

  memory_pool->AllocateHalfTensor(ptr_sz, mem_pool_sz);  // in: index 0
  memory_pool->AllocateHalfTensor(ptr_sz, mem_pool_sz);  // out: index 1
  memory_pool->AllocateHalfTensor(G * C, mem_pool_sz);  // gamma: index 2
  memory_pool->AllocateHalfTensor(G * C, mem_pool_sz);  // beta: index 3

"""
)

SHAPE_EVAL_TEMPLATE = jinja2.Template(
    """
    """
)

EXEC_TEMPLATE = jinja2.Template(
    """
    C = C / G;
    std::vector<ck::index_t> i_inStrides;

    i_inStrides.push_back(H * W * G * C);
    i_inStrides.push_back(W * G * C);
    i_inStrides.push_back(G * C);
    i_inStrides.push_back(C);
    i_inStrides.push_back(1);

    std::vector<ck::index_t> gamma_beta_Strides;
    gamma_beta_Strides.push_back(0);
    gamma_beta_Strides.push_back(0);
    gamma_beta_Strides.push_back(0);
    gamma_beta_Strides.push_back(C);
    gamma_beta_Strides.push_back(1);

    auto device_instance = {{instance}}{};
    auto argument_ptr = device_instance.MakeArgumentPointer(
        {static_cast<ck::index_t>(N),
         static_cast<ck::index_t>(H),
         static_cast<ck::index_t>(W),
         static_cast<ck::index_t>(G),
         static_cast<ck::index_t>(C)},
        i_inStrides, // x stride
        gamma_beta_Strides,
        gamma_beta_Strides,
        i_inStrides, // y stride
        std::vector<ck::index_t>{0, 0},
        std::vector<ck::index_t>{0, 0},
        {1, 2, 4}, // reduction dimension: [H, W, C]
        1e-5,
        static_cast<{{dtype}} *>(input),
        static_cast<{{dtype}} *>(gamma),
        static_cast<{{dtype}} *>(beta),
        static_cast<{{dtype}} *>(output),
        nullptr,
        nullptr,
        YElementOp{}
    );

    if(!device_instance.IsSupportedArgument(argument_ptr.get()))
    {
        LOG(FATAL) << "wrong! " << device_instance.GetTypeString() << " with the specified compilation parameters does not support this Groupnorm problem.";
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
                   int64_t N,
                   int64_t H,
                   int64_t W,
                   int64_t G,
                   int64_t C,
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
{{indent}}   {{N}},
{{indent}}   {{H}},
{{indent}}   {{W}},
{{indent}}   {{G}},
{{indent}}   {{C}},
{{indent}}   stream
{{indent}});
    """
)


PROFILER_FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}   {{input}},
{{indent}}   {{gamma}},
{{indent}}   {{beta}},
{{indent}}   {{output}},
{{indent}}   N,
{{indent}}   H,
{{indent}}   W,
{{indent}}   G,
{{indent}}   C,
{{indent}}   stream
{{indent}});
    """
)


@registry.reg("rocm.groupnorm.config")
def groupnorm_extract_config(func_attrs):
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

    def f_proc_op(op: ck_lib.groupnorm_operation.GroupNormOperation):
        if op.In == data_type and op.Out == data_type and op.acc_dtype == acc_type:
            return op
        return None

    op_kind = ck_lib.library.OperationKind.GroupNorm
    extra_kind = 5
    extract_ops = list(Target.current()._operators[op_kind][extra_kind].items())
    groupnorm_ops = OrderedDict()
    for key, value in extract_ops:
        op = value[0]
        if f_proc_op(op) is not None:
            groupnorm_ops[key] = op
    func_attrs["op_instance"] = groupnorm_ops


def get_func_signature_profiler(func_attrs: Dict[str, Any]) -> str:
    return FUNC_SIGNATURE.render(
        func_name=func_attrs["name"],
        dtype="void",
        input_ndim=5,
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
    int64_t N,
    int64_t H,
    int64_t W,
    int64_t G,
    int64_t C,
    hipStream_t stream)
{
    C = C / G;

    std::vector<ck::index_t> i_inStrides = {
        static_cast<int>(H * W * G * C),
        static_cast<int>(W * G * C),
        static_cast<int>(G * C),
        static_cast<int>(C),
        1
    };

    std::vector<ck::index_t> gamma_beta_Strides = {
        0, 0, 0, static_cast<int>(C), 1
    };

    auto argument_ptr = device_instance.MakeArgumentPointer(
        {static_cast<ck::index_t>(N),
         static_cast<ck::index_t>(H),
         static_cast<ck::index_t>(W),
         static_cast<ck::index_t>(G),
         static_cast<ck::index_t>(C)},
        i_inStrides,
        gamma_beta_Strides,
        gamma_beta_Strides,
        i_inStrides,
        std::vector<ck::index_t>{0, 0},
        std::vector<ck::index_t>{0, 0},
        {1, 2, 4},
        1e-5,
        memory_pool->RequestHalfTensorByIdx(0),
        memory_pool->RequestHalfTensorByIdx(2),
        memory_pool->RequestHalfTensorByIdx(3),
        memory_pool->RequestHalfTensorByIdx(1),
        nullptr,
        nullptr,
        YElementOp{}
    );

    if(!device_instance.IsSupportedArgument(argument_ptr.get()))
    {
        return -1;
    }

    // warmup
    auto invoker = device_instance.MakeInvokerPointer();
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
        in_0, in_1, in_2, in_3, in_4,
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


@registry.reg("rocm.groupnorm.gen_profiler")
def groupnorm_gen_profiler(
    func_attrs: Dict[str, Any],
    workdir: str,
    indent: str = "  ",
    use_swish: bool = False,
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
    use_swish : bool, optional
        Use swish if True
    """
    # N, H, W, C
    x = func_attrs["inputs"][0]
    spec = ROCMSpec()
    lib_dtype = spec.dtype_to_lib_type(x._attrs["dtype"])

    shapes = []
    for dim in func_attrs["inputs"][0]._attrs["shape"]:
        if isinstance(dim, IntImm):
            shapes.append(dim.value())
        else:
            shapes.append(dim.upper_bound())
    return gen_profiler(
        func_attrs,
        workdir,
        rank=5,
        extra_header_template=EXTRA_HEADERS,
        extra_code=EXTRA_CODE_TEMPLATE.render(use_swish=use_swish, dtype=lib_dtype),
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
    extra_code_template: jinja2.Template,
    get_func_signature: Any,
    use_swish: bool = False,
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
    extra_code_template : jinja2.Template
        Extra code template.

    Returns
    -------
    str
        The rendered template of generated function body.
    """
    x = func_attrs["inputs"][0]
    spec = ROCMSpec()
    lib_dtype = spec.dtype_to_lib_type(x._attrs["dtype"])

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

    exec_cond_template = func_attrs["exec_cond_template"]
    exec_paths = ""
    for key, _ in instances.items():
        fname = "f" + sha1(key.encode()).hexdigest()
        program = exec_template.render(instance=fname, dtype=lib_dtype)
        exec_inst = exec_cond_template.render(indent="  ", cond=key, program=program)
        exec_paths += exec_inst

    return norm_common.FUNC_TEMPLATE.render(
        instances_decl=instance_decl,
        func_signature=get_func_signature(func_attrs),
        shape_eval="",
        exec_paths=exec_paths,
        extra_headers=extra_header_template.render(),
        extra_code=extra_code_template.render(use_swish=use_swish, dtype=lib_dtype),
    )


@registry.reg("rocm.groupnorm.gen_function")
def groupnorm_gen_function(func_attrs: Dict[str, Any], use_swish: bool = False) -> str:
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
    # N, H, W, C
    shapes = func_attrs["inputs"][0]._attrs["shape"]

    return gen_function(
        func_attrs,
        SHAPE_EVAL_TEMPLATE,
        EXEC_TEMPLATE,
        EXTRA_HEADERS,
        EXTRA_CODE_TEMPLATE,
        get_func_signature,
        use_swish,
    )


def get_func_signature(func_attrs: Dict[str, Any]) -> str:
    input_ndim = func_attrs["inputs"][0]._rank()
    return FUNC_SIGNATURE.render(
        func_name=func_attrs["name"],
        dtype="void",
        input_ndim=input_ndim,
    ).strip()


@registry.reg("rocm.groupnorm.func_decl")
def groupnorm_gen_func_decl(func_attrs: Dict[str, Any]):
    return FUNC_DECL.render(func_signature=get_func_signature(func_attrs))


@registry.reg("rocm.groupnorm.func_call")
def groupnorm_gen_func_call(func_attrs, indent="  "):
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
    assert len(shapes) == 4, (
        f"GroupNorm only supports input with rank == 4, current rank: {len(shapes)}"
    )

    N = shapes[0]._attrs["name"]
    H = shapes[1]._attrs["name"]
    W = shapes[2]._attrs["name"]
    G = func_attrs["num_groups"]
    C = shapes[3]._attrs["name"]

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        input=input_name,
        gamma=gamma_name,
        beta=beta_name,
        output=output_name,
        N=N,
        H=H,
        W=W,
        G=G,
        C=C,
        indent=indent,
    )
