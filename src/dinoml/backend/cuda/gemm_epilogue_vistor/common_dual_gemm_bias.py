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
from functools import partial
from hashlib import sha1
from typing import Any, Dict

import jinja2

from dinoml.backend.backend_spec import CUDASpec
from dinoml.backend.common import gemm_common
from dinoml.backend.cuda.gemm_universal import common
from dinoml.backend.target import Target

from dinoml.utils import alignment
from dinoml.utils.cutlass_lib.gemm_operation import GemmOperation

KERNEL_KEY_TEMPLATE = jinja2.Template(
    """
cutlass_{{opcode_class_name}}_{{extended_name}}_{{threadblock}}_{{layout}}_align_{{align_ab}}_{{align_c}}
"""
)

TENSOR_DECL_TEMPLATE = jinja2.Template(
    """
  int64_t a_ptr_sz = 1;
{% for dim in adims %}
  a_ptr_sz *= {{dim}};
{% endfor %}

  int64_t b0_ptr_sz = 1;
{% for dim in bdims %}
  b0_ptr_sz *= {{dim}};
{% endfor %}

  int64_t b1_ptr_sz = b0_ptr_sz;
{% if broadcast_b1 %}
  // scale b1_ptr_sz down by the broadcasted dim
  b1_ptr_sz /= {{ bdims[broadcasted_bdim_id] }};
{% endif %}

  int64_t c_ptr_sz = 1;
{% for dim in cdims %}
  c_ptr_sz *= {{dim}};
{% endfor %}

  // The value 1 is used to force ptr_max_sz to be non-zero
  int64_t ptr_max_sz = std::max<int64_t>({1, a_ptr_sz, b0_ptr_sz, c_ptr_sz});
  size_t one_copy_sz = a_ptr_sz + b0_ptr_sz + c_ptr_sz;
  one_copy_sz += b1_ptr_sz;
  int64_t mem_pool_sz = memory_pool->ComputeMemPoolSize(one_copy_sz, ptr_max_sz, device_properties.l2CacheSize);

  memory_pool->AllocateTensor(a_ptr_sz, mem_pool_sz);  // a_ptr: index 0
  memory_pool->AllocateTensor(b0_ptr_sz, mem_pool_sz);  // b0_ptr: index 1
  memory_pool->AllocateTensor(c_ptr_sz, mem_pool_sz, /*is_output*/true);  // c_ptr: index 2
  memory_pool->AllocateTensor(b1_ptr_sz, mem_pool_sz);  // b1_ptr: index 3
  memory_pool->AllocateTensor(c_dim1, mem_pool_sz);  // bias0_ptr: index 4
  memory_pool->AllocateTensor(c_dim1, mem_pool_sz);  // bias1_ptr: index 5
"""
)

EXEC_TEMPLATE = jinja2.Template(
    """
//  TODO: cast to right dtype
//{{indent}}using ElementComputeEpilogue = typename {{instance}}::ElementAccumulator;
{{indent}}using ElementCompute = typename {{instance}}::DualGemmKernel::Epilogue0::OutputOp::ElementCompute;

{{indent}}using coord_t = cutlass::gemm::GemmCoord::Index;
{{indent}}typename {{instance}}::Arguments arguments{

{{problem_args}}

{{indent}}};
{% if is_profiler %}
{{indent}}size_t workspace_size = gemm_op.get_workspace_size(arguments);
{{indent}}cutlass::device_memory::allocation<uint8_t> local_workspace(workspace_size);

{{indent}}workspace = local_workspace.get();
{{indent}}GLOBAL_WORKSPACE_SIZE = workspace_size;
{% else %}
{{indent}}{{instance}} gemm_op;
{% endif %}

{{indent}} auto status = gemm_op.can_implement(arguments);
{{indent}}CUTLASS_CHECK(status);
{{indent}}status = gemm_op.initialize(arguments, workspace, stream);
{{indent}}CUTLASS_CHECK(status);
{{indent}}status = gemm_op(stream);
{{indent}}CUTLASS_CHECK(status);

{{indent}}return;
"""
)


SRC_TEMPLATE = jinja2.Template(
    """
#include <iostream>
#include <memory>
#include <random>
#include <vector>
#include <iostream>
#include "short_file.h"
#include <cuda_bf16.h>

#include "dinoml/cutlass_global_override.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/epilogue/thread/linear_combination_silu.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/device_memory.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"

using bfloat16 = nv_bfloat16;

{{extra_code}}

#define CUTLASS_CHECK(status)                                                         \\
  {                                                                                   \\
    cutlass::Status error = status;                                                   \\
    if (error != cutlass::Status::kSuccess) {                                         \\
      auto msg = std::string("[") + __SHORT_FILE__ + "] Got cutlass error: " +              \\
          cutlassGetStatusString(error) + " at: " + std::to_string(__LINE__);         \\
      std::cerr << msg << std::endl;                                                  \\
      throw std::runtime_error(msg);                                                  \\
    }                                                                                 \\
  }

{{instances}}

{% if is_profiler %}
template <typename GemmInstance>
void {{function_name}} (
    GemmInstance& gemm_op,
{% else %}
void {{function_name}} (
{% endif %}
    void* a_ptr,
    void* b0_ptr,
    void* b1_ptr,
    void* c_ptr,
    void* bias0_ptr,
    void* bias1_ptr,
    uint8_t* workspace,
{% if support_split_k %}
    int split_k,
{% endif %}
{% for idx in range(input_ndims) %}
    int64_t* a_dim{{idx}},
{% endfor %}
{% for idx in range(weight_ndims) %}
    int64_t* b_dim{{idx}},
{% endfor %}
{% for idx in range(input_ndims) %}
    int64_t* c_dim{{idx}},
{% endfor %}
  cudaStream_t stream
  ) {
  {{shape_eval}}
  {{input_addr_calculator}}
  {{output_addr_calculator}}
  {{extra_shape}}
  {{input_output_checks}}

  {{exec_paths}}
  {% for idx in range(input_ndims) %}
      std::cout << "input_ndims{{idx}}: " << *a_dim{{idx}} << std::endl;
  {% endfor %}
  {% for idx in range(weight_ndims) %}
      std::cout << "weight_ndims{{idx}}: " << *b_dim{{idx}} << std::endl;
  {% endfor %}
  {% for idx in range(input_ndims) %}
      std::cout << "output_ndims{{idx}}: " << *c_dim{{idx}} << std::endl;
  {% endfor %}
  throw std::runtime_error(
      "Unsupported workload for this {{function_name}} specialization."
  );
}
""",
    trim_blocks=True,
    lstrip_blocks=True,
)


FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{
{{indent}}{{local_dim_defs}}
{{indent}}{{func_name}}(
{% if is_profiler %}
{{indent}}    gemm_op,
{% endif %}
{{indent}}    {{a_ptr}},
{{indent}}    {{b0_ptr}},
{{indent}}    {{b1_ptr}},
{{indent}}    {{c_ptr}},
{{indent}}    {{bias0_ptr}},
{{indent}}    {{bias1_ptr}},
{{indent}}    global_workspace_,
{{indent}}    {{split_k}},
{% for dim in adims %}
{{indent}}    {{dim}},
{% endfor %}
{% for dim in bdims %}
{{indent}}    {{dim}},
{% endfor %}
{% for dim in cdims %}
{{indent}}    {{dim}},
{% endfor %}
{{indent}}    stream
{{indent}});
{{indent}}}
"""
)


FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
  void*,
  void*,
  void*,
  void*,
  void*,
  void*,
  uint8_t*,
{% if support_split_k %}
    int,
{% endif %}
{% for idx in range(input_ndims) %}
  int64_t*,
{% endfor %}
{% for idx in range(weight_ndims) %}
  int64_t*,
{% endfor %}
{% for idx in range(input_ndims) %}
  int64_t*,
{% endfor %}
  cudaStream_t
);
"""
)


# TODO Merge all alignment into single profiler
PROFILER_TEMPLATE = jinja2.Template(
    """
size_t GLOBAL_WORKSPACE_SIZE = 0;

#include <sstream>

{{op_func}}

template <typename DType>
struct ProfilerMemoryPool;

template <typename GemmInstance>
int benchmark_{{function_name}} (
    GemmInstance &gemm_op,
    const char *gemm_op_name,
    ProfilerMemoryPool<{{elem_type}}>* memory_pool,
    uint8_t* global_workspace_,
{% if support_split_k %}
    int split_k,
{% endif %}
{% for idx in range(input_ndims) %}
    int64_t* a_dim{{idx}},
{% endfor %}
{% for idx in range(weight_ndims) %}
    int64_t* b_dim{{idx}},
{% endfor %}
{% for idx in range(output_ndims) %}
    int64_t* c_dim{{idx}},
{% endfor %}
    cudaStream_t stream
  ) {
  // warmup
  for (int i = 0; i < 5; ++i) {
    {{func_call}}
  }
  cudaEvent_t events[2];
  for (auto & event : events) {
    cudaEventCreate(&event);
  }
  cudaEventRecord(events[0], stream);
  for (int i = 0; i < 10; ++i) {
    {{func_call}}
  }
  cudaEventRecord(events[1], stream);
  cudaEventSynchronize(events[1]);
  float runtime_ms = 0;
  cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
  for (auto event : events) {
    (void)cudaEventDestroy(event);
  }
  // TODO: output workspace
  if (runtime_ms < 0.00001) {
      throw std::runtime_error(
      "OOB in cutlass."
    );
  }
  std::cout << "OP:" << gemm_op_name << ",";
  std::cout << "TIME:" << runtime_ms << ",";
  std::cout << "WS:" << GLOBAL_WORKSPACE_SIZE << std::endl;
  return 0;
}

template <typename DType>
struct ProfilerMemoryPool {
  ProfilerMemoryPool() : shared_input_tensor(false) {
    std::random_device rd;
    gen = std::mt19937(rd());
    uniform_dist = std::uniform_int_distribution<int64_t>(1, 48964896);
    offsets.reserve(512);
    strides.reserve(512);
    copies.reserve(512);
    ptrs.reserve(512);
    blobs.reserve(512);
  }
  ~ProfilerMemoryPool() {}

  int64_t ComputeMemPoolSize(size_t one_copy_sz, size_t ptr_max_sz, size_t l2_cache_bytes) {
    int times_covers_l2_cache = (int)std::ceil(l2_cache_bytes / sizeof(DType) / ptr_max_sz);
    int64_t mem_pool_sz = std::max(2, std::min(512, times_covers_l2_cache));
    size_t free_global_mem = 0;
    size_t total_global_mem = 0;
    cudaError_t cuda_error = cudaMemGetInfo(&free_global_mem, &total_global_mem);
    if (cuda_error != cudaSuccess) {
      auto error_msg = std::string("Failed to invoke cudaMemGetInfo: ") +
          cudaGetErrorName(cuda_error) + ", at " + __SHORT_FILE__;
      throw std::runtime_error(error_msg);
    }
    size_t single_copy_nbytes = one_copy_sz * sizeof(DType);
    while (mem_pool_sz > 0) {
      size_t nbytes = single_copy_nbytes * mem_pool_sz;
      if (nbytes < free_global_mem) {
        break;
      }
      mem_pool_sz--;
    }

    if (mem_pool_sz <= 1) {
      size_t minimal_required_nbytes = ptr_max_sz * sizeof(DType);
      if (minimal_required_nbytes > free_global_mem) {
        // We absolutely run out of memory
        auto error_msg = std::string("no enough GPU memory: requested ") +
            std::to_string(minimal_required_nbytes) + ", available: " +
            std::to_string(free_global_mem) + ", ptr_max_sz: " +
            std::to_string(ptr_max_sz) + ", at " + __SHORT_FILE__;
        throw std::runtime_error(error_msg);
      } else {
        // Let's try to allocate a single blob that is large enough to hold
        // all input tensors. Note that this is still an approximation, because
        // we may still hit cudaErrorMemoryAllocation error while allocating
        // memory for the output. We will rely on cudaMalloc to throw out
        // an exception in such a case.
        shared_input_tensor = true;
        AllocateGaussianTensor(ptr_max_sz);
      }
      return 1;
    }
    return mem_pool_sz;
  }

  DType* AllocateGaussianTensor(int64_t size) {
    size_t length = size * sizeof(DType);
    blobs.emplace_back(length);
    DType* ptr = reinterpret_cast<DType*>(blobs.back().get());

    uint64_t seed = uniform_dist(gen);
    double mean = 0.f;
    double std = 1.f;

    cutlass::reference::device::BlockFillRandomGaussian(ptr, size, seed, mean,
                                                        std);

    return ptr;
  }

  int AllocateTensor(int64_t size, int64_t copy, bool is_output = false) {
    offsets.push_back(0);
    strides.push_back(size);
    copies.push_back(copy);
    DType *ptr;
    if (!is_output && shared_input_tensor) {
      ptr = reinterpret_cast<DType*>(blobs.back().get());
    } else {
      ptr = AllocateGaussianTensor(size * copy);
    }
    ptrs.push_back(reinterpret_cast<void*>(ptr));
    return ptrs.size() - 1;
  }

  DType* RequestTensorByIdx(int idx) {
    auto copy = copies.at(idx);
    auto offset = offsets.at(idx);
    auto stride = strides.at(idx);
    DType* ptr = reinterpret_cast<DType*>(ptrs.at(idx));
    ptr += offset;
    offset += stride;
    if (offset == copy * stride) {
        offset = 0;
    }
    offsets[idx] = offset;
    return ptr;
  }

  std::vector<int64_t> offsets;
  std::vector<int64_t> strides;
  std::vector<int64_t> copies;
  std::vector<void*> ptrs;
  std::vector<cutlass::DeviceAllocation<uint8_t> > blobs;
  std::mt19937 gen;
  std::uniform_int_distribution<int64_t> uniform_dist;
  // make a shared blob to hold all inputs in cases we do not have
  // enough GPU memory
  bool shared_input_tensor;
};


int main(int argc, char** argv) {
  int device_idx;
  cudaDeviceProp device_properties;
  cudaError_t result = cudaGetDevice(&device_idx);
  auto memory_pool = std::make_unique<ProfilerMemoryPool<{{elem_type}}>>();
  if (result != cudaSuccess) {
    std::ostringstream errorStream;
    errorStream << "cudaGetDevice() call failed! "
                << "Error code: " << cudaGetErrorName(result)
                << " Error message: " << cudaGetErrorString(result);
    throw std::runtime_error(errorStream.str());
  }

  result = cudaGetDeviceProperties(&device_properties, device_idx);

  if (result != cudaSuccess) {
    std::ostringstream errorStream;
    errorStream << "cudaGetDeviceProperties() call failed! "
                << "Error code: " << cudaGetErrorName(result)
                << " Error message: " << cudaGetErrorString(result);
    throw std::runtime_error(errorStream.str());
  }

  {{args_parse}}

  uint8_t* global_workspace_ = nullptr;
  cudaStream_t stream = nullptr;

  {{tensor_decl}}

  {{benchmark_instances}}
  return 0;
}
"""
)


INPUT_OUTPUT_CHECKS_TEMPLATE = jinja2.Template(
    """
  int64_t a_size = 1;
{% for idx in range(input_ndims) %}
    a_size *= *a_dim{{idx}};
{% endfor %}
  if (a_size != 0 && !a_ptr) {
    throw std::runtime_error("input a is null!");
  }

  int64_t b_size = 1;
{% for idx in range(weight_ndims) %}
    b_size *= *b_dim{{idx}};
{% endfor %}
  if (b_size != 0 && !b0_ptr) {
    throw std::runtime_error("input b is null!");
  }

  int64_t c_size = 1;
{% for idx in range(output_ndims) %}
    c_size *= *c_dim{{idx}};
{% endfor %}
  if (c_size != 0) {
    if (!c_ptr) {
      throw std::runtime_error("input c is null!");
    }
  } else {
    // output is empty and safe to return
    return;
  }

  // One of the input tensor are empty
  if (a_size == 0 || b_size == 0) {
    return;
  }
"""
)


def kernel_name(op, func_attrs):
    """Returns kernel_name given input cutlass op_instance and operator attrs."""

    import dinoml.utils.cutlass_lib as cutlass_lib

    threadblock = op.tile_description.procedural_name()
    extended_name = op.extended_name()
    opcode_class_name = cutlass_lib.library.OpcodeClassNames[
        op.tile_description.math_instruction.opcode_class
    ]
    layout = op.layout_name()
    align_ab = op.A.alignment
    align_c = op.C.alignment

    name = KERNEL_KEY_TEMPLATE.render(
        threadblock=threadblock,
        extended_name=extended_name,
        opcode_class_name=opcode_class_name,
        layout=layout,
        align_ab=align_ab,
        align_c=align_c,
    )
    return name.replace("\n", "")


def extract_config(f_proc_op, func_attrs):
    return common.extract_config(f_proc_op, partial(kernel_name, func_attrs=func_attrs))


def dual_gemm_instance(
    op_def: str, func_attrs: Dict[str, Any], for_profiler: bool
) -> str:
    tmp = op_def.replace(
        "GemmIdentityThreadblockSwizzle<8>", "GemmIdentityThreadblockSwizzle<1>"
    )
    return tmp


def emit_instance(
    op,
    for_profiler,
    f_instance_convertor=dual_gemm_instance,
    emit_kernel=False,
    func_attrs=None,
    broadcast_b1=False,
    has_bias=False,
):
    import dinoml.utils.cutlass_lib as cutlass_lib

    emiter = cutlass_lib.gemm_operation.EmitDualGemmInstance()
    op_def = emiter.emit(op, broadcast_b1=broadcast_b1, has_bias=has_bias)
    op_def = f_instance_convertor(op_def, func_attrs, for_profiler)
    return op_def


def default_fproc(
    *,
    op: GemmOperation,
    a_layout,
    b_layout,
    c_layout,
    epilogue_name,
    epilogue2_name,
    permute_layout=None,
    dtype="float16",
):
    import copy

    import dinoml.utils.cutlass_lib as cutlass_lib

    backend_spec = CUDASpec()
    data_type = backend_spec.dtype_to_lib_type(dtype)

    ret = []
    # skip simt kernels
    if (
        op.tile_description.math_instruction.opcode_class
        == cutlass_lib.library.OpcodeClass.Simt
    ):
        return ret

    if op.tile_description.stages < 3:
        return ret

    if data_type == "float":
        if (
            op.tile_description.math_instruction.element_a
            != cutlass_lib.library.DataType.f32
            and op.tile_description.math_instruction.element_a
            != cutlass_lib.library.DataType.tf32
        ):
            return ret
    acc_type = cutlass_lib.library.DataType.f32

    if (
        "no_tf32" in Target.current()._kwargs
        and data_type == "float"
        and Target.current()._kwargs["no_tf32"]
    ):
        if (
            op.tile_description.math_instruction.element_a
            == cutlass_lib.library.DataType.tf32
        ):
            return ret

    # check target use fp16 acc
    if "use_fp16_acc" in Target.current()._kwargs and data_type == "cutlass::half_t":
        if Target.current()._kwargs["use_fp16_acc"]:
            acc_type = cutlass_lib.library.DataType.f16

    if (
        cutlass_lib.library.DataTypeTag[op.A.element] == data_type
        and cutlass_lib.library.DataTypeTag[op.B.element] == data_type
        and cutlass_lib.library.DataTypeTag[op.C.element] == data_type
        and op.accumulator_type() == acc_type
        and op.A.layout == a_layout
        and op.B.layout == b_layout
    ):
        op = copy.deepcopy(op)
        # set output major
        op.C.layout = c_layout
        # set epilogue
        op.epilogue_functor = cutlass_lib.library.EpilogueFunctorName[epilogue_name]
        op.epilogue_functor2 = cutlass_lib.library.EpilogueFunctorName[epilogue2_name]
        op.element_epilogue = acc_type
        if permute_layout is not None:
            op.permute_layout = cutlass_lib.library.EpiloguePermuteLayoutName[
                permute_layout
            ]
        # set C alignment
        alignments = alignment.get_alignments(dtype)
        for i in alignments:
            op = copy.deepcopy(op)
            op.C.alignment = i
            ret.append(op)
    return ret


def make_fproc(
    func_attrs,
    layout,
    dtype="float16",
):
    """
    This function sets a callback for processing the epilogue of the kernel
    associated with func_attrs.
    """

    def fproc(op):
        a_layout, b_layout, c_layout = layout.cutlass_lib_layouts()
        return default_fproc(
            op=op,
            a_layout=a_layout,
            b_layout=b_layout,
            c_layout=c_layout,
            epilogue_name=func_attrs["epilogue"],
            epilogue2_name=func_attrs["epilogue2"],
            dtype=dtype,
        )

    func_attrs["op_instance"] = extract_config(fproc, func_attrs)


def gen_function(
    func_attrs,
    exec_cond_template,
    problem_args,
    input_ndims,
    weight_ndims,
    output_ndims,
    dim_info_dict,
    f_instance_convertor=dual_gemm_instance,
    emit_kernel=False,
    support_split_k=False,
    input_addr_calculator="",
    output_addr_calculator="",
    extra_code="",
    broadcast_b1=False,
):
    func_name = func_attrs["name"]
    exec_path = func_attrs["exec_path"]
    op_instance = func_attrs["op_instance"]
    inst_def_flag = set()
    instances = {}
    instance_decl = ""
    for exec_item in exec_path.values():
        fname = "f" + sha1(exec_item.exec_cond.encode()).hexdigest()
        algo = exec_item.algo
        if algo not in inst_def_flag:
            config = emit_instance(
                op_instance[algo],
                for_profiler=False,
                f_instance_convertor=f_instance_convertor,
                emit_kernel=emit_kernel,
                func_attrs=func_attrs,
                broadcast_b1=broadcast_b1,
                has_bias=True,
            )
            inst_def_flag.add(algo)
        else:
            config = ""
        inst = common.INSTANCE_TEMPLATE.render(
            config=config, name=fname, config_name=common.extract_config_name(config)
        )
        instances[exec_item.exec_cond] = inst
        instance_decl += inst
    shape_eval_func = gemm_common.gen_shape_eval_code(
        indent=1, dtype="int64_t", dim_info_dict=dim_info_dict, is_ptr=True
    )
    exec_paths = ""
    for key, _ in instances.items():
        fname = "f" + sha1(key.encode()).hexdigest()
        program = EXEC_TEMPLATE.render(
            indent="    ",
            instance=fname,
            problem_args=problem_args,
            support_split_k=support_split_k,
        )
        exec_inst = exec_cond_template.render(indent="  ", cond=key, program=program)
        exec_paths += exec_inst
    input_output_checks = INPUT_OUTPUT_CHECKS_TEMPLATE.render(
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        output_ndims=output_ndims,
    )

    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_lib_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )

    return SRC_TEMPLATE.render(
        instances=instance_decl,
        function_name=func_name,
        dtype=elem_input_type,
        shape_eval=shape_eval_func,
        input_addr_calculator=input_addr_calculator,
        output_addr_calculator=output_addr_calculator,
        input_output_checks=input_output_checks,
        exec_paths=exec_paths,
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        output_ndims=output_ndims,
        support_split_k=support_split_k,
        extra_code=extra_code,
    )


def gen_profiler(
    func_attrs,
    workdir,
    profiler_filename,
    dim_info_dict,
    src_template,
    problem_args_template,
    args_parser_template,
    emit_kernel=False,
    support_split_k=False,
    output_addr_calculator="",
    extra_code="",
    broadcast_b1=False,
    broadcasted_bdim_id=0,
    ndims=2,
):
    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_lib_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    elem_output_type = backend_spec.dtype_to_lib_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )
    elem_type = backend_spec.dtype_to_backend_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    op_type = func_attrs["op"]
    op_instance = func_attrs["op_instance"]

    adims = ["&a_dim" + str(i) for i in range(ndims)]
    bdims = ["&b_dim" + str(i) for i in range(ndims)]
    cdims = ["&c_dim" + str(i) for i in range(ndims)]
    shape_func = gemm_common.gen_shape_eval_code(
        indent=2, dtype="int64_t", dim_info_dict=dim_info_dict, is_ptr=True
    )
    instance_name_base = "GemmInstance"
    exec_program = EXEC_TEMPLATE.render(
        indent="  ",
        instance=instance_name_base,
        is_profiler=True,
        support_split_k=support_split_k,
        problem_args=problem_args_template.render(
            elem_input_type=elem_input_type,
            elem_output_type=elem_output_type,
            broadcast_b1=broadcast_b1,
        ),
    )
    input_output_checks = INPUT_OUTPUT_CHECKS_TEMPLATE.render(
        input_ndims=ndims,
        weight_ndims=ndims,
        output_ndims=ndims,
    )

    function_name = "gemm"
    instances = []
    benchmark_instances = []
    for instance_idx, (op_name, op) in enumerate(op_instance.items()):
        config = emit_instance(
            op,
            for_profiler=True,
            emit_kernel=emit_kernel,
            func_attrs=func_attrs,
            broadcast_b1=broadcast_b1,
        )
        config_name = common.extract_config_name(config)
        instance_name = f"{instance_name_base}_{instance_idx}"
        gemm_op = f"gemm_op_{instance_idx}"
        instance = common.INSTANCE_TEMPLATE.render(
            config_name=config_name, name=instance_name, config=config
        )
        benchmark_instance = common.BENCHMARK_INSTANCE_TEMPLATE.render(
            indent="  ",
            instance_name=instance_name,
            gemm_op=gemm_op,
            gemm_op_name=op_name,
            func_name=f"benchmark_{function_name}",
            support_split_k=support_split_k,
            split_k="split_k",
            adims=adims,
            bdims=bdims,
            cdims=cdims,
        )
        instances.append(instance)
        benchmark_instances.append(benchmark_instance)
    op_func = src_template.render(
        is_profiler=True,
        instances="\n".join(instances),
        function_name=function_name,
        input_ndims=ndims,
        weight_ndims=ndims,
        output_ndims=ndims,
        shape_eval=shape_func,
        input_output_checks=input_output_checks,
        exec_paths=exec_program,
        output_addr_calculator=output_addr_calculator,
        support_split_k=support_split_k,
        extra_code=extra_code,
    )
    benchmark_adims = ["a_dim" + str(i) for i in range(ndims)]
    benchmark_bdims = ["b_dim" + str(i) for i in range(ndims)]
    benchmark_cdims = ["c_dim" + str(i) for i in range(ndims)]
    func_call = FUNC_CALL_TEMPLATE.render(
        is_profiler=True,
        func_name=function_name,
        a_ptr="memory_pool->RequestTensorByIdx(0)",
        b0_ptr="memory_pool->RequestTensorByIdx(1)",
        b1_ptr="memory_pool->RequestTensorByIdx(3)",
        c_ptr="memory_pool->RequestTensorByIdx(2)",
        bias0_ptr="memory_pool->RequestTensorByIdx(4)",
        bias1_ptr="memory_pool->RequestTensorByIdx(5)",
        split_k="split_k",
        adims=benchmark_adims,
        bdims=benchmark_bdims,
        cdims=benchmark_cdims,
    )
    # TODO: Render args_parse by caller.
    args_parse = (
        args_parser_template
        if isinstance(args_parser_template, str)
        else args_parser_template.render()
    )
    code = PROFILER_TEMPLATE.render(
        op_func=op_func,
        support_split_k=support_split_k,
        args_parse=args_parse,
        function_name=function_name,
        input_ndims=ndims,
        weight_ndims=ndims,
        output_ndims=ndims,
        func_call=func_call,
        tensor_decl=TENSOR_DECL_TEMPLATE.render(
            adims=benchmark_adims,
            bdims=benchmark_bdims,
            cdims=benchmark_cdims,
            broadcast_b1=broadcast_b1,
            broadcasted_bdim_id=broadcasted_bdim_id,
        ),
        benchmark_instances="\n".join(benchmark_instances),
        elem_type=elem_type,
    )
    # FIXME: remove file_pairs once we have make -j ready for building
    # an entire graph
    file_pairs = []
    common.add_profiler(file_pairs, workdir, op_type, profiler_filename, code)
    # build
    return common.build_profiler(file_pairs)


def gen_function_call(func_attrs, indent="  "):
    a = func_attrs["inputs"][0]
    ashapes = func_attrs["input_accessors"][0].original_shapes
    b0 = func_attrs["inputs"][1]
    bshapes = func_attrs["input_accessors"][1].original_shapes
    b1 = func_attrs["inputs"][2]
    bias0 = func_attrs["inputs"][3]
    bias1 = func_attrs["inputs"][4]
    c = func_attrs["outputs"][0]
    cshapes = func_attrs["output_accessors"][0].original_shapes
    # overwrite the global defs if we have input TensorAccessor
    local_dim_defs = common.gen_local_dim_defs(func_attrs, indent=indent)
    adims = ["&" + dim._attrs["name"] for dim in ashapes]
    bdims = ["&" + dim._attrs["name"] for dim in bshapes]
    cdims = ["&" + dim._attrs["name"] for dim in cshapes]
    return FUNC_CALL_TEMPLATE.render(
        local_dim_defs=local_dim_defs,
        func_name=func_attrs["name"],
        a_ptr=a._attrs["name"],
        b0_ptr=b0._attrs["name"],
        b1_ptr=b1._attrs["name"],
        bias0_ptr=bias0._attrs["name"],
        bias1_ptr=bias1._attrs["name"],
        c_ptr=c._attrs["name"],
        split_k=func_attrs["split_k"],
        adims=adims,
        bdims=bdims,
        cdims=cdims,
        indent=indent,
    )
