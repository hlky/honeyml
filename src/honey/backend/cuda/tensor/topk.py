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
topk kernel codegen for CUDA.
"""

from typing import Any, Dict

from honey.backend import registry
from honey.backend.backend_spec import CUDASpec
from honey.backend.common.tensor import topk_common

# pylint: disable=C0301

header_files = """
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include <cub/cub.cuh>
#include <type_traits>

using bfloat16 = nv_bfloat16;

namespace cub {

namespace {

template <typename... Ts>
using void_t = void;

template <typename T, typename = void>
struct is_defined : std::false_type {};

// We rely on the fact that defined classes can be instantiated, and the size
// of their instance can be calculated. Therefore, if sizeof() fails, then
// SFINAE will fall back to previous option (i.e. std::false_type).
template <typename T>
struct is_defined<T, void_t<decltype(sizeof(T))>> : std::true_type {};

}  // namespace

struct ThrowAwayType {}; // don't care about this template type

// A forward declaration is needed in case this type doesn't exist yet.
template <> struct NumericTraits<bfloat16>;

using NumericTraitsType = std::conditional_t<is_defined<NumericTraits<bfloat16>>::value, ThrowAwayType, bfloat16>;

template <> struct NumericTraits<NumericTraitsType>
  : BaseTraits<FLOATING_POINT, true, false, unsigned short, bfloat16> {};

template<> struct Traits<NumericTraitsType>
  : NumericTraits<bfloat16> {};

}  // namespace cub

"""


@registry.reg("cuda.topk.gen_function")
def topk_gen_function(func_attrs: Dict[str, Any]) -> str:
    return topk_common.gen_function(func_attrs, header_files, CUDASpec())


@registry.reg("cuda.topk.func_decl")
def topk_gen_function_decl(func_attrs: Dict[str, Any]):
    return topk_common.gen_function_decl(func_attrs, CUDASpec())


@registry.reg("cuda.topk.func_call")
def topk_gen_function_call(func_attrs, indent="  "):
    return topk_common.gen_function_call(func_attrs, CUDASpec(), indent)


@registry.reg("cuda.topk.gen_profiler")
def gen_profiler(func_attrs, workdir):
    return topk_common.gen_profiler(func_attrs, workdir, header_files, CUDASpec())
