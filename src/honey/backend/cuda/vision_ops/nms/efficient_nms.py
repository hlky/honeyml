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
efficient_nms kernel codegen for CUDA.
"""

from typing import Any, Dict

from honey.backend import registry
from honey.backend.backend_spec import CUDASpec
from honey.backend.common.vision_ops import efficient_nms_common

# pylint: disable=C0301

func_header_files = """
#include <cuda_fp16.h>
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cuda_runtime_api.h"
#include "cub/cub.cuh"
"""

profiler_header_files = """
#include <cuda_fp16.h>
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include <cub/cub.cuh>
"""


@registry.reg("cuda.efficient_nms.gen_function")
def efficient_nms_gen_function(func_attrs: Dict[str, Any]) -> str:
    return efficient_nms_common.gen_function(func_attrs, func_header_files, CUDASpec())


@registry.reg("cuda.efficient_nms.func_decl")
def efficient_nms_gen_function_decl(func_attrs: Dict[str, Any]):
    return efficient_nms_common.gen_function_decl(func_attrs, CUDASpec())


@registry.reg("cuda.efficient_nms.func_call")
def efficient_nms_gen_function_call(func_attrs, indent="  "):
    return efficient_nms_common.gen_function_call(func_attrs, CUDASpec(), indent)


@registry.reg("cuda.efficient_nms.gen_profiler")
def gen_profiler(func_attrs, workdir):
    return efficient_nms_common.gen_profiler(
        func_attrs, workdir, profiler_header_files, CUDASpec()
    )
