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
Elementwise codegen for CUDA.
"""

import os
from typing import Any, Dict

from honey.backend import registry
from honey.backend.backend_spec import CUDASpec
from honey.backend.common import elementwise_common
from honey.backend.target import Target

HEAD_TEMPLATE = """
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/constants.h"
#include "cutlass/epilogue/thread/activation.h"
#include "math_constants.h"
#include "kernels/cuda/custom_math.cuh"
"""


@registry.reg("cuda.fused_elementwise.gen_function")
def fused_elementwise_gen_function(func_attrs: Dict[str, Any]) -> str:
    """Generates fused_elementwise function definition."""
    return elementwise_common.fused_elementwise_gen_function(
        func_attrs=func_attrs,
        head_template=HEAD_TEMPLATE,
        backend_spec=CUDASpec(),
    )


@registry.reg("cuda.fused_elementwise.func_decl")
def fused_elementwise_gen_function_decl(func_attrs):
    """Generates fused_elementwise function declaration."""
    return elementwise_common.fused_elementwise_gen_function_decl(
        func_attrs=func_attrs,
        backend_spec=CUDASpec(),
    )


@registry.reg("cuda.fused_elementwise.func_call")
def fused_elementwise_gen_function_call(func_attrs, indent):
    """Generates fused_elementwise function call."""
    return elementwise_common.fused_elementwise_gen_function_call(
        func_attrs=func_attrs,
        indent=indent,
        backend_spec=CUDASpec(),
    )
