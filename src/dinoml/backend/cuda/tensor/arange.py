from typing import Any, Dict

from dinoml.backend import registry
from dinoml.backend.backend_spec import CUDASpec
from dinoml.backend.common.tensor.arange_common import (
    gen_function,
    gen_function_call,
    gen_function_decl,
)

CUDA_HEADER_FILES = """
#include <cuda_fp16.h>
#include <cuda_runtime.h>
"""


@registry.reg("cuda.arange.gen_function")
def cuda_arange_gen_function(func_attrs: Dict[str, Any]) -> str:
    return gen_function(func_attrs, CUDA_HEADER_FILES, CUDASpec())


@registry.reg("cuda.arange.func_decl")
def cuda_arange_gen_function_decl(func_attrs: Dict[str, Any]) -> str:
    return gen_function_decl(func_attrs, CUDASpec())


@registry.reg("cuda.arange.func_call")
def cuda_arange_gen_function_call(func_attrs: Dict[str, Any], indent="  ") -> str:
    return gen_function_call(func_attrs, indent)
