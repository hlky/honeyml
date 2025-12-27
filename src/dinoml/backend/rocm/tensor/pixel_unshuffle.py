from dinoml.backend import registry
from dinoml.backend.backend_spec import ROCMSpec
from dinoml.backend.common import pixel_unshuffle_common


Header_Files = """
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include "library/include/ck/library/utility/host_tensor.hpp"
"""


@registry.reg("rocm.pixel_unshuffle.gen_function")
def gen_function(
    func_attrs,
    template_path,
):
    func_name = func_attrs["name"]
    x = func_attrs["inputs"][0]
    output = func_attrs["outputs"][0]
    backend_spec = ROCMSpec()
    input_type = backend_spec.dtype_to_backend_type(x._attrs["dtype"])
    output_type = backend_spec.dtype_to_backend_type(output._attrs["dtype"])
    return pixel_unshuffle_common.SRC_TEMPLATE.render(
        header_files=Header_Files,
        function_name=func_name,
        index_type=backend_spec.index_type,
        prefix=backend_spec.prefix,
        elem_input_type=input_type,
        elem_output_type=output_type,
    )


@registry.reg("rocm.pixel_unshuffle.func_decl")
def pixel_unshuffle_gen_function_decl(func_attrs):
    return pixel_unshuffle_common.gen_function_decl(func_attrs, backend_spec=ROCMSpec())


@registry.reg("rocm.pixel_unshuffle.func_call")
def pixel_unshuffle_gen_function_call(func_attrs, indent="    "):
    return pixel_unshuffle_common.gen_function_call(
        func_attrs, backend_spec=ROCMSpec(), indent=indent
    )
