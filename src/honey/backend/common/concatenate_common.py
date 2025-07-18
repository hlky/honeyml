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
backend concatenate function common templates.
"""
from copy import deepcopy
from typing import List

import jinja2

from honey.backend.common import tensor_accessor_codegen
from honey.backend.target import Target

from honey.compiler.base import IntImm
from honey.compiler.ops.tensor import concatenate
from honey.utils.shape_utils import is_empty_rank1_tensor

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
    void * /*output*/,
    {{index_type}} *[] /*output_shape*/,
    const void *[] /*inputs*/,
    const {{index_type}} *[], /* real_input_shapes, representing shapes of those inputs
                                 whose masks are False, i.e. inputs that will be
                                 copied to the output tensor by concat.*/
    const {{index_type}} *[], /* all_input_shapes, including both kinds of inputs,
                                 i.e. not matter input_mask being True or False */
    const bool [] /*input_masks*/,
    const {{index_type}} [] /*concat_dim_sizes*/,
    {{index_type}} /*concat_dim*/,
    {{index_type}} /*rank*/,
    {{index_type}} /*num_real_inputs*/,
    {{index_type}} /*num_all_inputs*/,
    {{prefix}}Stream_t
);
"""
)


KERNEL_SRC_TEMPLATE = jinja2.Template(
    """
#include <assert.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "short_file.h"
#include "kernels/tensor_accessor.cuh"

{{header_src}}

#ifndef CHECK_ERROR_CAT
#define CHECK_ERROR_CAT(expr)                                \\
  do {                                                       \\
    {{prefix}}Error_t status = (expr);                       \\
    if (status != {{prefix}}Success) {                       \\
      auto msg = std::string("Got error: ") +                \\
        {{prefix}}GetErrorString(status) +                   \\
        " at " + __SHORT_FILE__ + ": " + std::to_string(__LINE__); \\
      std::cerr << msg << std::endl;                         \\
      throw std::runtime_error(msg);                         \\
    }                                                        \\
  } while (0)
#endif // CHECK_ERROR_CAT

#ifndef LAUNCH_CHECK_CAT
#define LAUNCH_CHECK_CAT() CHECK_ERROR_CAT({{prefix}}GetLastError())
#endif // LAUNCH_CHECK_CAT

{% if element_func_def %}
{{element_func_def}}
{% endif %}

namespace {

template <typename T, {{index_type}} NumInputs>
struct InputMetaData {
  const T *inputs[NumInputs]; /* pointer to each input */
  TensorAccessor input_accessors[NumInputs];
  int64_t concat_dim_offsets[NumInputs]; /* offset of each input along
                                            the concat dimension */
  int64_t concat_dim_values[NumInputs]; /* concat dimension value of
                                           each input */
  int64_t num_elems[NumInputs]; /* number of elements of each input */
};

template <{{index_type}} Rank>
struct OutputMetaData {
  int64_t output_shape[Rank];
  int64_t output_strides[Rank];
};

__host__ __device__ __forceinline__
int64_t get_num_elems(const {{index_type}} *shape, {{index_type}} rank) {
  int64_t num = 1;
  for ({{index_type}} i = 0; i < rank; i++) {
    num *= shape[i];
  }
  return num;
}

template <typename INDEX_T, {{index_type}} Rank>
__host__ __device__ int64_t compute_output_elem_offset(
    const int64_t *output_shape,
    const int64_t *output_strides,
    const INDEX_T input_concat_dim_value,
    const INDEX_T concat_dim,
    INDEX_T linear_idx) {
  INDEX_T offset = 0;
  for (INDEX_T i = Rank - 1; i >= 1; --i) {
    INDEX_T cur_dim_size =
        i == concat_dim ? input_concat_dim_value : output_shape[i];
    INDEX_T next_dim_idx = linear_idx / cur_dim_size;
    INDEX_T cur_dim_idx = linear_idx - cur_dim_size * next_dim_idx;
    INDEX_T cur_dim_offset = cur_dim_idx * static_cast<INDEX_T>(output_strides[i]);
    offset += cur_dim_offset;
    linear_idx = next_dim_idx;
  }
  return offset + linear_idx * static_cast<INDEX_T>(output_strides[0]);
}
} // namespace

template <typename READ_T, typename ELEM_T, typename INDEX_T, {{index_type}} Rank,
          {{index_type}} NumInputs, {{index_type}} ElemsPerThread>
__global__ void
concatenate_kernel(
    ELEM_T *orig_output,
    OutputMetaData<Rank> output_meta,
    InputMetaData<ELEM_T, NumInputs> input_meta,
    const INDEX_T concat_dim,
    const INDEX_T output_concat_dim_stride) {
  const INDEX_T tid = blockIdx.x * blockDim.x + threadIdx.x;
  const INDEX_T block_y = blockIdx.y % NumInputs;
  READ_T* output = reinterpret_cast<READ_T*>(orig_output);

  READ_T* input = const_cast<READ_T*>(
      reinterpret_cast<const READ_T*>(input_meta.inputs[block_y]));
  const TensorAccessor &input_accessor = input_meta.input_accessors[block_y];
  INDEX_T input_offset = input_meta.concat_dim_offsets[block_y];
  INDEX_T num_input_elems = input_meta.num_elems[block_y];
  INDEX_T input_concat_dim_value = input_meta.concat_dim_values[block_y];
  INDEX_T output_offset = input_offset * output_concat_dim_stride;

  constexpr unsigned read_t_sz = sizeof(READ_T);
  constexpr unsigned elem_t_sz = sizeof(ELEM_T);
  static_assert(read_t_sz >= elem_t_sz && (read_t_sz % elem_t_sz == 0));
  constexpr INDEX_T n_of_elem_t = read_t_sz / elem_t_sz;
  // number of READ_T elements per thread
  INDEX_T reads_per_thread_in_read_t = ElemsPerThread / n_of_elem_t;
  const INDEX_T num_elems_in_read_t = num_input_elems / n_of_elem_t;
  INDEX_T read_idx = tid;

#pragma unroll
  for (INDEX_T i = 0; i < reads_per_thread_in_read_t;
       i++, read_idx += blockDim.x * gridDim.x) {
    if (read_idx >= num_elems_in_read_t) {
      break;
    }
    READ_T tmp_v = *(input_accessor.get<ELEM_T, READ_T>(input, read_idx));
    /* make sure to adjust read_idx, which refers to location at
       (read_idx * n_of_elem_t) actually */

    INDEX_T output_elem_offset =
        compute_output_elem_offset<INDEX_T, Rank>(output_meta.output_shape,
                                                  output_meta.output_strides,
                                                  input_concat_dim_value,
                                                  concat_dim,
                                                  read_idx * n_of_elem_t);
    {% if element_func %}
    output[(output_offset + output_elem_offset) / n_of_elem_t] = {{element_func}}(tmp_v);
    {% else %}
    output[(output_offset + output_elem_offset) / n_of_elem_t] = tmp_v;
    {% endif %}
  }
}

enum class LoadVecType {
  VT_HALF = 0,
  VT_BFLOAT16,
  VT_FLOAT,
  VT_FLOAT2,
  VT_FLOAT4
};

template <typename ELEM_T>
static inline LoadVecType get_vec_type({{index_type}} dim_size) {
  {{index_type}}  size_elem_t = sizeof(ELEM_T);

#define HANDLE_ONE_VEC_TYPE(load_vec_type, vec_type)  \\
  if (sizeof(vec_type) % size_elem_t == 0) {          \\
    {{index_type}}  n_of_elem_t = sizeof(vec_type) / size_elem_t; \\
    if (dim_size % n_of_elem_t == 0) {                \\
      return load_vec_type;                           \\
    }                                                 \\
  }

  HANDLE_ONE_VEC_TYPE(LoadVecType::VT_FLOAT4, float4)
  HANDLE_ONE_VEC_TYPE(LoadVecType::VT_FLOAT2, float2)
  HANDLE_ONE_VEC_TYPE(LoadVecType::VT_FLOAT, float)
  if constexpr (std::is_same_v<ELEM_T, half>) {
    HANDLE_ONE_VEC_TYPE(LoadVecType::VT_HALF, half)
  }
  if constexpr (std::is_same_v<ELEM_T, bfloat16>) {
    HANDLE_ONE_VEC_TYPE(LoadVecType::VT_BFLOAT16, bfloat16)
  }
#undef HANDLE_ONE_VEC_TYPE
  throw std::runtime_error(
      "Cannot resolve LoadVecType."
  );
}

template <typename ELEM_T, typename INDEX_T, {{index_type}} Rank, {{index_type}} NumInputs,
          {{index_type}} ElemsPerThread, {{index_type}} ThreadsPerBlock>
void concatenate_kernel_launcher(
    void *output,
    const {{index_type}} *output_shape,
    const void *inputs[],
    const {{index_type}} *real_input_shapes[],
    const TensorAccessor *input_accessors[],
    const int64_t concat_dim_offsets[],
    const {{index_type}} concat_dim,
    LoadVecType min_vec_type,
    {{prefix}}Stream_t stream) {

  OutputMetaData<Rank> output_meta;
  output_meta.output_strides[Rank - 1] = 1;
  output_meta.output_shape[Rank - 1] = output_shape[Rank - 1];
  for (INDEX_T i = Rank - 2; i >= 0; i--) {
    output_meta.output_strides[i] =
        output_meta.output_strides[i+1] * output_shape[i+1];
    output_meta.output_shape[i] = output_shape[i];
  }

  InputMetaData<ELEM_T, NumInputs> input_meta;
  INDEX_T max_num_input_elems = 0;
  for (INDEX_T i = 0; i < NumInputs; i++) {
    INDEX_T num_elems = get_num_elems(real_input_shapes[i], Rank);
    input_meta.inputs[i] = static_cast<const ELEM_T*>(inputs[i]);
    input_meta.input_accessors[i] = *(input_accessors[i]);
    input_meta.concat_dim_offsets[i] = concat_dim_offsets[i];
    input_meta.concat_dim_values[i] = real_input_shapes[i][concat_dim];
    input_meta.num_elems[i] = num_elems;

    max_num_input_elems = num_elems > max_num_input_elems ?
                          num_elems : max_num_input_elems;
  }

  constexpr INDEX_T elems_per_block = ThreadsPerBlock * ElemsPerThread;
  INDEX_T m = (max_num_input_elems % elems_per_block != 0);
  INDEX_T num_blocks_x =
      (max_num_input_elems / elems_per_block) + m;
  dim3 grid_config = dim3(static_cast<unsigned>(num_blocks_x), NumInputs);

#define HANDLE_ONE_VEC_TYPE(load_vec_type, vec_type)                        \\
    if (min_vec_type == load_vec_type) {                                    \\
      if (ElemsPerThread * sizeof(ELEM_T) < sizeof(vec_type)) {             \\
         throw std::runtime_error(                                          \\
           std::string("No valid kernel available for ") + #vec_type);      \\
      }                                                                     \\
      concatenate_kernel<vec_type, ELEM_T, INDEX_T, Rank, NumInputs, ElemsPerThread> \\
        <<<grid_config, ThreadsPerBlock, 0, stream>>>(                      \\
            static_cast<ELEM_T*>(output),                                   \\
            output_meta,                                                    \\
            input_meta,                                                     \\
            concat_dim,                                                     \\
            output_meta.output_strides[concat_dim]);                        \\
      LAUNCH_CHECK_CAT();                                                   \\
      return;                                                               \\
    }

    HANDLE_ONE_VEC_TYPE(LoadVecType::VT_FLOAT4, float4)
    HANDLE_ONE_VEC_TYPE(LoadVecType::VT_FLOAT2, float2)
    HANDLE_ONE_VEC_TYPE(LoadVecType::VT_FLOAT, float)
    if constexpr (std::is_same_v<ELEM_T, half>) {
      HANDLE_ONE_VEC_TYPE(LoadVecType::VT_HALF, half)
    }
    if constexpr (std::is_same_v<ELEM_T, bfloat16>) {
      HANDLE_ONE_VEC_TYPE(LoadVecType::VT_BFLOAT16, bfloat16)
    }

#undef HANDLE_ONE_VEC_TYPE
  throw std::runtime_error("Invalid LoadVecType\\n");
}

#undef CHECK_ERROR_CAT
#undef LAUNCH_CHECK_CAT
"""
)


DUMMY_KERNEL_TEMPLATE = jinja2.Template(
    """
#include <assert.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
{{header_src}}

void {{func_name}}(
    void *output,
    {{index_type}} *output_shape[],
    const void *inputs[],
    const {{index_type}} *real_input_shapes[],
    const {{index_type}} *all_input_shapes[],
    const bool input_masks[],
    const {{index_type}} concat_dim_sizes[],
    {{index_type}} concat_dim,
    {{index_type}} rank,
    {{index_type}} num_real_inputs,
    {{index_type}} num_all_inputs,
    {{prefix}}Stream_t stream
    ) {
  // DO NOTHING
}
"""
)


INPUT_ACCESSOR_DEFS_TEMPLATE = jinja2.Template(
    """
{{input_accessors}}

{{indent}}const TensorAccessor *input_accessors[{{num_real_inputs}}] = {

{{indent}}  {{input_accessor_refs}}

{{indent}}};
"""
)


EXEC_COND_TEMPLATE = jinja2.Template(
    """
{{indent}}if (rank == {{rank}} && num_real_inputs == {{num_real_inputs}}) {

{{input_accessor_defs}}

{{indent}}  LoadVecType min_vec_type = LoadVecType::VT_FLOAT4;
{{indent}}  int64_t accessor_idx = 0;
{{indent}}  for ({{index_type}} i = 0; i < num_all_inputs; i++) {
{{indent}}    int local_alignment;
{{indent}}    if (!input_masks[i] ||
{{indent}}        input_accessors[accessor_idx]->stride_dim == -1) {
{{indent}}      local_alignment = all_input_shapes[i][rank - 1];
{{indent}}      // int64_t is ok here because this happens on CPU
{{indent}}      for (int64_t j = rank - 2; j >= concat_dim; j--) {
{{indent}}        local_alignment *= all_input_shapes[i][j];
{{indent}}      }
{{indent}}    } else {
{{indent}}      local_alignment =
{{indent}}          input_accessors[accessor_idx]->max_alignment();
{{indent}}    }
{{indent}}    LoadVecType vec_type = get_vec_type<{{elem_type}}>(local_alignment);
{{indent}}    min_vec_type = vec_type < min_vec_type ? vec_type : min_vec_type;
{{indent}}    if (input_masks[i]) {
{{indent}}      accessor_idx++;
{{indent}}    }
{{indent}}  }

{{indent}}  {{index_type}} local_output_shape[] = {
{% for idx in range(rank - 1) %}
{{indent}}    *(output_shape[{{idx}}]),
{% endfor %}
{{indent}}    *(output_shape[{{rank - 1}}])
{{indent}}  };

{{indent}}/* TODO: more profiling on ElemsPerThread and ThreadsPerBlock */
{{indent}}if (use_int32_index_math) {
{{indent}}  concatenate_kernel_launcher<{{elem_type}},
{{indent}}                    int32_t,
{{indent}}                    {{rank}}/*Rank*/,
{{indent}}                    {{num_real_inputs}}/*NumInputs*/,
{{indent}}                    {{elems_per_thread}}/*ElemsPerThread*/,
{{indent}}                    {{threads_per_block}}/*THREADS_PER_BLOCK*/>(
{{indent}}    output, local_output_shape, inputs, real_input_shapes, input_accessors,
{{indent}}    concat_dim_offsets.data(), concat_dim, min_vec_type, stream);
{{indent}}} else {
{{indent}}  concatenate_kernel_launcher<{{elem_type}},
{{indent}}                    int64_t,
{{indent}}                    {{rank}}/*Rank*/,
{{indent}}                    {{num_real_inputs}}/*NumInputs*/,
{{indent}}                    {{elems_per_thread}}/*ElemsPerThread*/,
{{indent}}                    {{threads_per_block}}/*THREADS_PER_BLOCK*/>(
{{indent}}    output, local_output_shape, inputs, real_input_shapes, input_accessors,
{{indent}}    concat_dim_offsets.data(), concat_dim, min_vec_type, stream);
{{indent}}}
{{indent}}return;
{{indent}}}
"""
)


SRC_TEMPLATE = jinja2.Template(
    """
{{kernel_src}}

void {{func_name}}(
    void *output,
    {{index_type}} *output_shape[],
    const void *inputs[],
    const {{index_type}} *real_input_shapes[], /* real_input_shapes, representing
                                 shapes of those inputs whose masks are False,
                                 i.e. inputs that will be copied to the output
                                 tensor by concat.*/
    const {{index_type}} *all_input_shapes[], /* all_input_shapes include both
                                 kinds of inputs, i.e. no matter input_mask being
                                 True or False */
    const bool input_masks[],
    const {{index_type}} concat_dim_sizes[],
    {{index_type}} concat_dim,
    {{index_type}} rank,
    {{index_type}} num_real_inputs,
    {{index_type}} num_all_inputs,
    {{prefix}}Stream_t stream
    ) {

  if (rank <= 0) {
    throw std::runtime_error("rank must be larger than 0!");
  }
  if (concat_dim >= rank) {
    throw std::runtime_error("concat_dim must be smaller than rank!");
  }
  if (num_real_inputs < 1) {
    throw std::runtime_error("the number of inputs must >= 1!");
  }


  for ({{index_type}} i = 0; i < rank; i++) {
    if (i == concat_dim) continue;
    {{index_type}} dim = real_input_shapes[0][i];

    for ({{index_type}} j = 1; j < num_real_inputs; j++) {
      if (real_input_shapes[j][i] != dim) {
        throw std::runtime_error(
          "invalid input shape, func_name: {{func_name}}, dim: " +
          std::to_string(dim) + ", input_shape: " +
          std::to_string(real_input_shapes[j][i])
        );
      }
    }
  }

  {{index_type}} output_concat_dim_value = 0;
  std::vector<int64_t> concat_dim_offsets;

  for ({{index_type}} i = 0; i < num_all_inputs; i++) {
    if (input_masks[i]) {
      concat_dim_offsets.push_back(output_concat_dim_value);
    }
    output_concat_dim_value += concat_dim_sizes[i];
  }
  for ({{index_type}} i = 0; i < rank; i++) {
    if (i == concat_dim) {
      *(output_shape[i]) = output_concat_dim_value;
    } else {
      *(output_shape[i]) = real_input_shapes[0][i];
    }
  }

  // If all input tensors are empty we are done
  bool empty = false;
  bool use_int32_index_math = true;
  for (int i = 0; i < num_real_inputs; i++) {
    int64_t num_elems = get_num_elems(real_input_shapes[i], rank);
    if (get_num_elems(real_input_shapes[i], rank) != 0) {
      empty = false;
      // make sure input is valid for each non-zero-size tensor
      if (!inputs[i]) {
        throw std::runtime_error("NULL input is found at: " + std::to_string(i));
      }
    }
    if (input_masks[i]) {
      use_int32_index_math &= can_use_32bit_index_math(num_elems);
    }
  }

  if (empty) {
    return;
  }

  // if the output has any zero dim size, we are done
  for (int i = 0; i < rank; i++) {
    if (*output_shape[i] == 0)
      return;
  }
  // make sure output is valid
  if (!output) {
    throw std::runtime_error("output is NULL!");
  }

{{exec_paths}}

  throw std::runtime_error(
      "Unsupported concat kernel specialization!"
  );
}
"""
)


INPUT_SHAPE_DEF_TEMPLATE = jinja2.Template(
    """
{{indent}}{{index_type}} {{input_shape_name}}[] = {
{{indent}}  {{input_dims}}
{{indent}}};
"""
)


INITIALIZATION_LOOP_TEMPLATE = jinja2.Template(
    """
{{indent}}  for(int {{var_name}}_idx_ = {{start_idx}}; {{var_name}}_idx_ < {{end_idx}}; {{var_name}}_idx_++) {
{{indent}}      {{var_name}}[{{var_name}}_idx_] = {{init_val}};
{{indent}}  }
"""
)


FUNC_CALL_TEMPLATE_OPT = jinja2.Template(
    """
{{indent}}{

{{indent}}  const void *inputs[{{num_inputs}}];
{{indent}}  {{inputs_initialization}}

{{real_input_shape_defs}}

{{indent}}  const {{index_type}} *real_input_shapes[{{num_real_input_shapes}}];
{{indent}}  {{real_input_shapes_initialization}}

{{all_input_shape_defs}}

{{indent}}  const {{index_type}} *all_input_shapes[{{num_all_input_shapes}}];
{{indent}}  {{all_input_shapes_initialization}}

{{indent}}  {{index_type}} *{{output}}_shape[] = {
{{indent}}    {{output_dim_refs}}
{{indent}}  };

{{indent}}  {{index_type}} concat_dim_sizes[{{num_concat_dim_sizes}}];
{{indent}}  {{concat_dim_sizes_initialization}}

{{indent}}  bool input_masks[{{num_input_masks}}];
{{indent}}  {{input_masks_initialization}}

{{indent}}  {{func_name}}(
{{indent}}      {{output_ptr}},
{{indent}}      {{output}}_shape,
{{indent}}      inputs,
{{indent}}      real_input_shapes,
{{indent}}      all_input_shapes,
{{indent}}      input_masks,
{{indent}}      concat_dim_sizes,
{{indent}}      {{concat_dim}}/*concat_dim*/,
{{indent}}      {{rank}}/*rank*/,
{{indent}}      {{num_real_inputs}}/*num_real_inputs*/,
{{indent}}      {{num_all_inputs}}/*num_all_inputs*/,
{{indent}}      stream
{{indent}}  );
{{indent}}}
"""
)


FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{

{{indent}}  const void *inputs[] = {
{{indent}}    {{inputs}}
{{indent}}  };

{{real_input_shape_defs}}

{{indent}}  const {{index_type}} *real_input_shapes[] = {
{{indent}}    {{real_input_shapes}}
{{indent}}  };

{{all_input_shape_defs}}

{{indent}}  const {{index_type}} *all_input_shapes[] = {
{{indent}}    {{all_input_shapes}}
{{indent}}  };

{{indent}}  {{index_type}} *{{output}}_shape[] = {
{{indent}}    {{output_dim_refs}}
{{indent}}  };

{{indent}}  {{index_type}} concat_dim_sizes[] = {
{{indent}}    {{concat_dim_sizes}}
{{indent}}  };

{{indent}}  bool input_masks[] = {
{{indent}}    {{input_masks}}
{{indent}}  };

{{indent}}  {{func_name}}(
{{indent}}      {{output_ptr}},
{{indent}}      {{output}}_shape,
{{indent}}      inputs,
{{indent}}      real_input_shapes,
{{indent}}      all_input_shapes,
{{indent}}      input_masks,
{{indent}}      concat_dim_sizes,
{{indent}}      {{concat_dim}}/*concat_dim*/,
{{indent}}      {{rank}}/*rank*/,
{{indent}}      {{num_real_inputs}}/*num_real_inputs*/,
{{indent}}      {{num_all_inputs}}/*num_all_inputs*/,
{{indent}}      stream
{{indent}}  );
{{indent}}}
"""
)


def gen_function_decl(func_attrs, backend_spec):
    """Generate function declaration.

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        Stores the operation attributes.
    Returns
    -------
    str
        Rendered function declaration.
    """
    return FUNC_DECL_TEMPLATE.render(
        func_name=func_attrs["name"],
        index_type=backend_spec.index_type,
        prefix=backend_spec.prefix,
    )


def gen_function(
    func_attrs,
    backend_spec,
    element_func=None,
    element_func_def=None,
):
    """Generates function body.

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        Stores the operation attributes.
    index_type: str
        Index type.
    prefix: str
        Backend function prefix, hip/cuda
    dtype_to_backend_type: Dict[str, str]
    header_src_template: jinja Template
    Header src template.

    Returns
    -------
    str
        Rendered function body.
    """
    inputs = func_attrs["inputs"]
    original_inputs = func_attrs["original_inputs"]
    concatenate.check_rank(original_inputs, func_attrs["concat_dim"])
    orig_x, _ = concatenate.get_first_non_empty_input_if_any(original_inputs)
    y = func_attrs["outputs"][0]
    x_shape = orig_x._attrs["shape"]

    input_type = backend_spec.dtype_to_backend_type(orig_x._attrs["dtype"])
    output_type = backend_spec.dtype_to_backend_type(y._attrs["dtype"])

    # TODO: support type cast
    if input_type != output_type:
        raise NotImplementedError("input type must equal to output type")

    def _stride(shape, dim):
        stride = 1
        for v in shape[dim:]:
            stride = stride * v._attrs["values"][0]
        return stride

    concat_dim = func_attrs["concat_dim"]
    assert concat_dim < len(x_shape)
    strides = [_stride(i._attrs["shape"], concat_dim) for i in inputs]
    # the max number of elements in each concat loop iteration
    elems_per_iter = max(strides) if len(strides) > 0 else 0
    threads_per_block = 1024
    # minimal number of elems per thread is 8, max is 480
    elems_per_thread = (
        480  # min(480, (int((elems_per_iter / threads_per_block + 8) / 8) * 8))
    )

    input_accessors = []
    input_accessor_refs = []
    for i in range(len(inputs)):
        accessor_name = f"input_accessor{i}"
        input_accessor_refs.append(f"&{accessor_name}")
        input_accessors.append(
            tensor_accessor_codegen.TENSOR_ACCESSOR_TEMPLATE.render(
                name=accessor_name, tensor_accessor=func_attrs["input_accessors"][i]
            )
        )
    input_accessor_defs = INPUT_ACCESSOR_DEFS_TEMPLATE.render(
        indent="    ",
        input_accessors="".join(input_accessors),
        num_real_inputs=len(inputs),
        input_accessor_refs=", ".join(input_accessor_refs),
    )

    # TODO: consider to add profiling paths for tuning
    # elems_per_thread and threads_per_block
    exec_paths = EXEC_COND_TEMPLATE.render(
        indent="  ",
        rank=len(x_shape),
        num_real_inputs=len(inputs),
        input_accessor_defs=input_accessor_defs,
        elem_type=input_type,
        elems_per_thread=elems_per_thread,
        threads_per_block=threads_per_block,
        index_type=backend_spec.index_type,
    )

    header_src = backend_spec.header_src_template.render()
    if len(inputs) > 0:
        kernel_src = KERNEL_SRC_TEMPLATE.render(
            element_func=element_func,
            element_func_def=element_func_def,
            header_src=header_src,
            index_type=backend_spec.index_type,
            prefix=backend_spec.prefix,
        )
        return SRC_TEMPLATE.render(
            kernel_src=kernel_src,
            func_name=func_attrs["name"],
            exec_paths=exec_paths,
            index_type=backend_spec.index_type,
            prefix=backend_spec.prefix,
        )

    return DUMMY_KERNEL_TEMPLATE.render(
        func_name=func_attrs["name"],
        header_src=header_src,
        index_type=backend_spec.index_type,
        prefix=backend_spec.prefix,
    )


def _make_initialization_loops(init_var: str, init_values: List[str]):
    def _make_loop_ranges(init_values: List[str]):
        """
        For the given list, return a list of tuples, where each tuple contains
        (start_idx, end_idx, val). It means that in the input list, all the elements
        between [start_idx, end_idx) have the same value 'val'
        """
        start_idx = 0
        res = []
        for end_idx in range(1, len(init_values)):
            if init_values[end_idx] == init_values[end_idx - 1]:
                continue
            res.append((start_idx, end_idx, init_values[start_idx]))
            start_idx = end_idx
        if len(init_values) > 0:
            res.append((start_idx, len(init_values), init_values[start_idx]))

        return res

    loop_ranges = _make_loop_ranges(init_values)
    loop_range_strs = []
    for start_idx, end_idx, val in loop_ranges:
        loop_range_strs.append(
            INITIALIZATION_LOOP_TEMPLATE.render(
                var_name=init_var,
                start_idx=start_idx,
                end_idx=end_idx,
                init_val=val,
            )
        )
    return "\n".join(loop_range_strs)


def gen_function_call(
    func_attrs,
    backend_spec,
    indent="  ",
):
    """Generates function call.

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        Stores the operation attributes.
    backend_spec : BackendSpec
        CUDA / RocM type definitions
    indent : str, optional
        Indent for template, by default "  ".

    Returns
    -------
    str
        Rendered function call.
    """
    inputs = func_attrs["inputs"]
    input_accessors = func_attrs["input_accessors"]
    assert len(inputs) == len(input_accessors), (
        "expected inputs and input_accessors to have the same length, but got: "
        f'{len(inputs)}, {len(input_accessors)}, op: {func_attrs["name"]}'
    )
    original_inputs = func_attrs["original_inputs"]
    concatenate.check_rank(original_inputs, func_attrs["concat_dim"])
    orig_x, _ = concatenate.get_first_non_empty_input_if_any(original_inputs)
    y = func_attrs["outputs"][0]
    concat_dim = func_attrs["concat_dim"]

    input_names = [i._attrs["name"] for i in inputs]
    real_input_shape_defs = []
    real_input_shape_names = []
    # It's not uncommon that multiple shape definitions share the same
    # dimension values. In such a case, we could keep a single definition.
    # In some rare cases, this little trick can dramatically reduce the
    # number of lines generated for the relevant concatenate op and thus
    # may improve the compilation time. Currently, we only enable this
    # optimization for cases where we care about compilation time, as
    # for most cases, the unoptimized version can generate more readable
    # code while having little impact to the compilation time.
    seen_input_shape_dims = {}
    input_shape_name_subs = {}
    optimize_for_compilation_time = Target.current()._kwargs.get(
        "optimize_for_compilation_time", False
    )

    def _make_dims_key(dims):
        dim_vals = [
            str(d.value()) if isinstance(d, IntImm) else d._attrs["name"] for d in dims
        ]
        return ",".join(dim_vals)

    non_empty_input, non_empty_idx = concatenate.get_first_non_empty_input_if_any(
        inputs
    )
    non_empty_input_accessor = input_accessors[non_empty_idx]
    for idx, (i, input_accessor) in enumerate(zip(inputs, input_accessors)):
        input_shape_name = f'{i._attrs["name"]}_shape_{idx}'
        orig_input_shape = input_accessor.original_shapes
        if is_empty_rank1_tensor(orig_input_shape):
            orig_dim = orig_input_shape[0]
            orig_input_shape = deepcopy(non_empty_input_accessor.original_shapes)
            orig_input_shape[concat_dim] = orig_dim
        dims = ", ".join([dim._attrs["name"] for dim in orig_input_shape])
        dims_key = _make_dims_key(orig_input_shape)
        seen_shape_name = seen_input_shape_dims.get(dims_key, None)
        if not optimize_for_compilation_time or seen_shape_name is None:
            one_shape_def = INPUT_SHAPE_DEF_TEMPLATE.render(
                indent="      ",
                input_shape_name=input_shape_name,
                input_dims=dims,
                index_type=backend_spec.index_type,
            )
            real_input_shape_defs.append(one_shape_def)
            real_input_shape_names.append(input_shape_name)
            seen_input_shape_dims[dims_key] = input_shape_name
        else:
            real_input_shape_names.append(seen_shape_name)
            input_shape_name_subs[input_shape_name] = seen_shape_name

    y_shape = y._attrs["shape"]
    y_dim_refs = ", ".join(["&" + dim._attrs["name"] for dim in y_shape])

    input_masks = func_attrs["input_masks"]
    input_indices = [idx for idx, m in enumerate(input_masks) if m is True]
    assert len(inputs) == len(input_indices)
    concat_dim_sizes = []
    for idx, mask in enumerate(input_masks):
        if is_empty_rank1_tensor(original_inputs[idx]._attrs["shape"]):
            d = "0"
        elif mask:
            d = "-1"
        else:
            d = str(original_inputs[idx]._attrs["shape"][concat_dim].value())
        concat_dim_sizes.append(d)

    # update dim size for real inputs
    for input_accessor, input_index in zip(input_accessors, input_indices):
        if is_empty_rank1_tensor(input_accessor.original_shapes):
            dim = input_accessor.original_shapes[0]._attrs["name"]
        else:
            dim = input_accessor.original_shapes[concat_dim]._attrs["name"]
        concat_dim_sizes[input_index] = dim

    input_mask_values = ["true" if mask is True else "false" for mask in input_masks]

    # all input shape defs and names, including those that are masked out
    all_input_shape_defs = []
    all_input_shape_names = []
    seen_input_shape_dims = {}
    # first, create shape defs for inputs that have been masked off
    for (
        mask,
        orig_input,
    ) in zip(input_masks, original_inputs):
        if mask is False:
            orig_input_shape_name = f'orig_{orig_input._attrs["name"]}_shape'
            if orig_input_shape_name not in all_input_shape_names:
                dims = ", ".join(
                    [str(dim._attrs["values"][0]) for dim in orig_input._attrs["shape"]]
                )
                dims_key = _make_dims_key(orig_input._attrs["shape"])
                seen_shape_name = seen_input_shape_dims.get(dims_key, None)
                if not optimize_for_compilation_time or seen_shape_name is None:
                    one_shape_def = INPUT_SHAPE_DEF_TEMPLATE.render(
                        indent="      ",
                        input_shape_name=orig_input_shape_name,
                        input_dims=dims,
                        index_type=backend_spec.index_type,
                    )
                    all_input_shape_defs.append(one_shape_def)
                    seen_input_shape_dims[dims_key] = orig_input_shape_name
                    all_input_shape_names.append(orig_input_shape_name)
                else:
                    all_input_shape_names.append(seen_shape_name)
            else:
                all_input_shape_names.append(orig_input_shape_name)
        else:
            all_input_shape_names.append("")
    # update all_input_shapes with real input shapes
    for idx, (input_tensor, input_index) in enumerate(zip(inputs, input_indices)):
        input_shape_name = f'{input_tensor._attrs["name"]}_shape_{idx}'
        sub_name = input_shape_name_subs.get(input_shape_name, None)
        if sub_name is not None:
            input_shape_name = sub_name
        all_input_shape_names[input_index] = input_shape_name

    if optimize_for_compilation_time:
        inputs_initialization = _make_initialization_loops("inputs", input_names)
        real_input_shapes_initialization = _make_initialization_loops(
            "real_input_shapes", real_input_shape_names
        )
        all_input_shapes_initialization = _make_initialization_loops(
            "all_input_shapes", all_input_shape_names
        )
        concat_dim_sizes_initialization = _make_initialization_loops(
            "concat_dim_sizes", concat_dim_sizes
        )
        input_masks_initialization = _make_initialization_loops(
            "input_masks", input_mask_values
        )

        # A special version that aims for reducing compilation time for some uncommon
        # cases, e.g. where we have more than 100 concatenate ops, each of which taking
        # thousands of real inputs.
        return FUNC_CALL_TEMPLATE_OPT.render(
            indent=indent,
            num_inputs=len(input_names),
            inputs_initialization=inputs_initialization,
            real_input_shape_defs="".join(real_input_shape_defs),
            num_real_input_shapes=len(real_input_shape_names),
            real_input_shapes_initialization=real_input_shapes_initialization,
            all_input_shape_defs="".join(all_input_shape_defs),
            num_all_input_shapes=len(all_input_shape_names),
            all_input_shapes_initialization=all_input_shapes_initialization,
            num_input_masks=len(input_mask_values),
            input_masks_initialization=input_masks_initialization,
            num_concat_dim_sizes=len(concat_dim_sizes),
            concat_dim_sizes_initialization=concat_dim_sizes_initialization,
            output_dim_refs=y_dim_refs,
            func_name=func_attrs["name"],
            output=y._attrs["name"],
            output_ptr=y._attrs["name"],
            concat_dim=concat_dim,
            rank=len(orig_x._attrs["shape"]),
            num_real_inputs=len(inputs),
            num_all_inputs=len(original_inputs),
            index_type=backend_spec.index_type,
        )
    else:
        return FUNC_CALL_TEMPLATE.render(
            indent=indent,
            inputs=",\n      ".join(input_names),
            real_input_shape_defs="".join(real_input_shape_defs),
            real_input_shapes=", ".join(real_input_shape_names),
            all_input_shape_defs="".join(all_input_shape_defs),
            all_input_shapes=", ".join(all_input_shape_names),
            input_masks=", ".join(input_mask_values),
            concat_dim_sizes=", ".join(concat_dim_sizes),
            output_dim_refs=y_dim_refs,
            func_name=func_attrs["name"],
            output=y._attrs["name"],
            output_ptr=y._attrs["name"],
            concat_dim=concat_dim,
            rank=len(orig_x._attrs["shape"]),
            num_real_inputs=len(inputs),
            num_all_inputs=len(original_inputs),
            index_type=backend_spec.index_type,
        )
