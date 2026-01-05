from typing import Any, Dict

import jinja2

from dinoml.backend import registry
from dinoml.backend.backend_spec import CUDASpec


SRC_TEMPLATE = jinja2.Template(
    r"""
#include <dinoml/device.h>
#include <ops/fir_downsample2d.h>

void {{function_name}}(
    void* out,
    const void* in,
    int N, int H, int W, int C,
    dinoml::DeviceStream stream
) {
    invoke_fir_downsample2d<{{elem_type}}>(out, in, N, H, W, C, stream);
}
"""
)


FUNC_DECL_TEMPLATE = jinja2.Template(
    r"""
void {{func_name}}(
    void*,
    const void*,
    int, int, int, int,
    dinoml::DeviceStream
);
"""
)


FUNC_CALL_TEMPLATE = jinja2.Template(
    r"""
{{indent}}{{func_name}}(
{{indent}}    {{output}},
{{indent}}    {{input}},
{{indent}}    {{N}},
{{indent}}    {{H}},
{{indent}}    {{W}},
{{indent}}    {{C}},
{{indent}}    stream
{{indent}});
"""
)


def _dtype_to_backend(dtype) -> str:
    return CUDASpec().dtype_to_backend_type(dtype)


def gen_function(func_attrs: Dict[str, Any]) -> str:
    func_name = func_attrs["name"]
    elem_type = _dtype_to_backend(func_attrs["outputs"][0]._attrs["dtype"])
    return SRC_TEMPLATE.render(function_name=func_name, elem_type=elem_type)


def gen_decl(func_attrs: Dict[str, Any]) -> str:
    return FUNC_DECL_TEMPLATE.render(func_name=func_attrs["name"])


def gen_call(func_attrs: Dict[str, Any], indent="  ") -> str:
    x = func_attrs["inputs"][0]
    y = func_attrs["outputs"][0]
    N, H, W, C = x._attrs["shape"]
    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=y._attrs["name"],
        input=x._attrs["name"],
        N=N._attrs["name"],
        H=H._attrs["name"],
        W=W._attrs["name"],
        C=C._attrs["name"],
        indent=indent,
    )


@registry.reg("cuda.fir_downsample2d.gen_function")
def cuda_fir_downsample2d_gen_function(func_attrs):
    return gen_function(func_attrs)


@registry.reg("cuda.fir_downsample2d.func_decl")
def cuda_fir_downsample2d_func_decl(func_attrs):
    return gen_decl(func_attrs)


@registry.reg("cuda.fir_downsample2d.func_call")
def cuda_fir_downsample2d_func_call(func_attrs, indent="  "):
    return gen_call(func_attrs, indent)
