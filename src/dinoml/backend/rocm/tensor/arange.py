from typing import Any, Dict

from dinoml.backend import registry
from dinoml.backend.backend_spec import ROCMSpec
from dinoml.backend.common.tensor.arange_common import (
    gen_function,
    gen_function_call,
    gen_function_decl,
)

HIP_HEADER_FILES = """
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
"""


@registry.reg("rocm.arange.gen_function")
def rocm_arange_gen_function(func_attrs: Dict[str, Any]) -> str:
    return gen_function(func_attrs, HIP_HEADER_FILES, ROCMSpec())


@registry.reg("rocm.arange.func_decl")
def rocm_arange_gen_function_decl(func_attrs: Dict[str, Any]) -> str:
    return gen_function_decl(func_attrs, ROCMSpec())


@registry.reg("rocm.arange.func_call")
def rocm_arange_gen_function_call(func_attrs: Dict[str, Any], indent="  ") -> str:
    return gen_function_call(func_attrs, indent)
