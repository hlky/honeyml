from dinoml.backend import registry
from dinoml.backend.backend_spec import CUDASpec
from dinoml.backend.common import meshgrid_common

Header_Files = """
#include <array>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "cutlass/util/host_tensor.h"

using bfloat16 = __nv_bfloat16;
"""


@registry.reg("cuda.meshgrid.gen_function")
def gen_function(func_attrs, template_path):
    func_name = func_attrs["name"]
    backend_spec = CUDASpec()
    x = func_attrs["inputs"][0]
    input_type = backend_spec.dtype_to_backend_type(x._attrs["dtype"])
    num_inputs = len(func_attrs["inputs"])
    return meshgrid_common.SRC_TEMPLATE.render(
        header_files=Header_Files,
        function_name=func_name,
        index_type=backend_spec.index_type,
        prefix=backend_spec.prefix,
        elem_input_type=input_type,
        elem_output_type=input_type,
        num_inputs=num_inputs,
    )


@registry.reg("cuda.meshgrid.func_decl")
def meshgrid_gen_function_decl(func_attrs):
    return meshgrid_common.gen_function_decl(func_attrs, backend_spec=CUDASpec())


@registry.reg("cuda.meshgrid.func_call")
def meshgrid_gen_function_call(func_attrs, indent="    "):
    return meshgrid_common.gen_function_call(
        func_attrs, backend_spec=CUDASpec(), indent=indent
    )
