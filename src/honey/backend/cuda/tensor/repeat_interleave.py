from typing import Any, Dict

from honey.backend import registry
from honey.backend.backend_spec import CUDASpec
from honey.backend.common.tensor.repeat_interleave_common import (
    gen_function,
    gen_function_call,
    gen_function_decl,
)

CUDA_HEADER_FILES = """
#include <array>
#include <cuda_fp16.h>
"""


@registry.reg("cuda.repeat_interleave.gen_function")
def cuda_repeat_interleave_gen_function(func_attrs: Dict[str, Any]) -> str:
    return gen_function(func_attrs, CUDA_HEADER_FILES, CUDASpec())


@registry.reg("cuda.repeat_interleave.func_decl")
def cuda_repeat_interleave_gen_function_decl(func_attrs: Dict[str, Any]) -> str:
    return gen_function_decl(func_attrs, CUDASpec())


@registry.reg("cuda.repeat_interleave.func_call")
def cuda_repeat_interleave_gen_function_call(
    func_attrs: Dict[str, Any], indent="  "
) -> str:
    return gen_function_call(func_attrs, indent)
