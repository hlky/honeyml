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
GEMM Specialization for C = fast_gelu(GeMM(A, B) + bias)
where A[RowMajor][M, K], B[ColMajor][N, K], bias[RowMajor][K], C[RowMajor][M, N]
"""
import jinja2

from honey.backend import registry

from honey.backend.backend_spec import CUDASpec
from honey.backend.cuda.gemm_universal import (
    common,
    common_bias_activation,
    common_no_bias,
)
from honey.backend.cuda.gemm_universal.layout import RCR


# pylint: disable=C0103,C0415,W0613,C0301,R1705,R1703

EXTRA_CODE = jinja2.Template(
    """
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/constants.h"
#include "cutlass/complex.h"
#include "cutlass/array.h"
#include "cutlass/half.h"
#include "cutlass/functional.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination_generic.h"

namespace cutlass {
namespace epilogue {
namespace thread {

template <
  typename ElementOutput_,                             ///< Data type used to load and store tensors
  int Count,                                           ///< Number of elements computed per operation
                                                       ///< Usually it is 128/sizeof_bits<ElementOutput_>,
                                                       ///< but we use 64 or 32 sometimes when there are not enough data to store
  typename ElementAccumulator_ = ElementOutput_,       ///< Accumulator data type
  typename ElementCompute_ = ElementOutput_,           ///< Data type used to compute linear combination
  ScaleType::Kind Scale = ScaleType::Default,          ///< Control Alpha and Beta scaling
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest
>
using LinearCombinationFastGELU = LinearCombinationGeneric<GELU_taylor, ElementOutput_, Count, ElementAccumulator_,
                                                          ElementCompute_, Scale, Round, true>;

} // namespace thread
} // namespace epilogue
} // namespace cutlass

"""
)


PROBLEM_ARGS_TEMPLATE = jinja2.Template(
    """
    cutlass::gemm::GemmUniversalMode::kGemm,                 // GemmUniversalMode mode
    cutlass::gemm::GemmCoord{
        static_cast<coord_t>(M),
        static_cast<coord_t>(N),
        static_cast<coord_t>(K)
    },                                                       // GemmCoord problem_size
    split_k,                                                 // int batch_count
    {ElementComputeEpilogue(1), ElementComputeEpilogue(0)},  // typename EpilogueOutputOp::Params epilogue
    ({{elem_input_type}}*) a_ptr,                            // void const * ptr_A
    ({{elem_input_type}}*) b_ptr,                            // void const * ptr_B
    nullptr,                                                 // void const * ptr_C
    ({{elem_output_type}}*) (c_ptr) + output_offset,         // void * ptr_D
    M * K,                                                   // int64_t batch_stride_A
    N * K,                                                   // int64_t batch_stride_B
    N,                                                       // int64_t batch_stride_C
    M * N,                                                   // int64_t batch_stride_D
    K,                                                       // typename LayoutA::Stride::LongIndex lda
    K,                                                       // typename LayoutB::Stride::LongIndex ldb
    0,                                                       // typename LayoutC::Stride::LongIndex ldc
    output_stride,                                           // typename LayoutC::Stride::LongIndex ldd
"""
)


PROBLEM_ARGS_TEMPLATE_CUTLASS_3X = jinja2.Template(
    """
    cutlass::gemm::GemmUniversalMode::kGemm,                     // GemmUniversalMode mode
    {
        static_cast<coord_t>(M),
        static_cast<coord_t>(N),
        static_cast<coord_t>(K),
        static_cast<coord_t>(1)
    },                                                           // ProblemShape problem_shape
    ({{elem_input_type}}*) a_ptr,                                // ElementA const* ptr_A
    {K, cute::Int<1>{}, cute::Int<0>{}},                         // StrideA dA
    ({{elem_input_type}}*) b_ptr,                                // ElementB const* ptr_B
    {K, cute::Int<1>{}, cute::Int<0>{}},                         // StrideB dB
    {
        {ElementComputeEpilogue(1), ElementComputeEpilogue(0)},  // typename ThreadEpilogueOp::Params thread
        ({{elem_output_type}}*) (c_ptr) + output_offset,         // ElementC const* ptr_C
        {cute::Int<0>{}, cute::Int<1>{}, cute::Int<0>{}},        // StrideC dC
        ({{elem_output_type}}*) (c_ptr) + output_offset,         // ElementD const* ptr_D
        {output_stride, cute::Int<1>{}, cute::Int<0>{}},         // StrideD dD
    },                                                           // EpilogueArguments epilogue
"""
)


@registry.reg("cuda.gemm_rcr_fast_gelu.config")
def gemm_rcr_config(func_attrs, dtype="float16"):
    common.make_fproc(func_attrs, RCR, include_cutlass_3x_ops=True)


@registry.reg("cuda.gemm_rcr_fast_gelu.gen_profiler")
def gen_profiler(func_attrs, workdir, profiler_filename, dim_info_dict):
    return common_bias_activation.gen_profiler(
        func_attrs=func_attrs,
        workdir=workdir,
        profiler_filename=profiler_filename,
        dim_info_dict=dim_info_dict,
        problem_args_template=PROBLEM_ARGS_TEMPLATE,
        problem_args_template_cutlass_3x=PROBLEM_ARGS_TEMPLATE_CUTLASS_3X,
        extra_code=EXTRA_CODE.render(),
    )


@registry.reg("cuda.gemm_rcr_fast_gelu.gen_function")
def gen_function(
    func_attrs,
    exec_cond_template,
    dim_info_dict,
):
    input_ndims = len(func_attrs["input_accessors"][0].original_shapes)
    weight_ndims = len(func_attrs["input_accessors"][1].original_shapes)
    output_ndims = len(func_attrs["output_accessors"][0].original_shapes)
    backend_spec = CUDASpec()
    elem_input_type = backend_spec.dtype_to_lib_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    elem_output_type = backend_spec.dtype_to_lib_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )
    problem_args = PROBLEM_ARGS_TEMPLATE.render(
        elem_input_type=elem_input_type,
        elem_output_type=elem_output_type,
    )
    problem_args_cutlass_3x = PROBLEM_ARGS_TEMPLATE_CUTLASS_3X.render(
        elem_input_type=elem_input_type,
        elem_output_type=elem_output_type,
    )
    return common.gen_function(
        func_attrs=func_attrs,
        src_template=common_no_bias.SRC_TEMPLATE,
        exec_cond_template=exec_cond_template,
        problem_args=problem_args,
        problem_args_cutlass_3x=problem_args_cutlass_3x,
        input_ndims=input_ndims,
        weight_ndims=weight_ndims,
        output_ndims=output_ndims,
        dim_info_dict=dim_info_dict,
        support_split_k=True,
        output_addr_calculator=common.OUTPUT_ADDR_CALCULATOR.render(
            stride_dim="N",
            output_accessor=func_attrs["output_accessors"][0],
        ),
        extra_code=EXTRA_CODE.render(),
    )


@registry.reg("cuda.gemm_rcr_fast_gelu.func_decl")
def gen_function_decl(func_attrs):
    return common_bias_activation.gen_function_decl(func_attrs)


@registry.reg("cuda.gemm_rcr_fast_gelu.func_call")
def gen_function_call(func_attrs, indent="  "):
    return common.gen_function_call(func_attrs, indent, bias_ptr_arg="nullptr")


@registry.reg("cuda.gemm_rcr_fast_gelu.filter")
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
