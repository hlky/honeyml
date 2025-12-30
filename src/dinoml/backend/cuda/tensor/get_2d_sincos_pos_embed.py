from typing import Any, Dict

import jinja2

from dinoml.backend import registry
from dinoml.backend.backend_spec import CUDASpec
from dinoml.compiler.base import IntImm, IntVar


SRC_TEMPLATE = jinja2.Template(
    r"""
#include <dinoml/device.h>
#include <ops/positional_embeddings.h>

void {{function_name}}(
    void* out,
    int embed_dim,
    int grid_h,
    int grid_w,
    int cls_token,
    int extra_tokens,
    float interpolation_scale,
    int base_size,
    dinoml::DeviceStream stream
) {
    dinoml::invoke_sincos_pos_embed_2d<{{elem_output_type}}>(
        out,
        embed_dim,
        grid_h,
        grid_w,
        cls_token,
        extra_tokens,
        interpolation_scale,
        base_size,
        stream
    );
}
"""
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    r"""
void {{func_name}}(
    void*,
    int,
    int,
    int,
    int,
    int,
    float,
    int,
    dinoml::DeviceStream
);
"""
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    r"""
{{indent}}{{func_name}}(
{{indent}}    {{output}},
{{indent}}    {{embed_dim}},
{{indent}}    {{grid_h}},
{{indent}}    {{grid_w}},
{{indent}}    {{cls_token}},
{{indent}}    {{extra_tokens}},
{{indent}}    {{interpolation_scale}},
{{indent}}    {{base_size}},
{{indent}}    stream
{{indent}});
"""
)


def _to_int_expr(x) -> str:
    if isinstance(x, IntImm):
        return str(x.value())
    if isinstance(x, IntVar):
        return x._attrs["name"]
    if isinstance(x, bool):
        return "1" if x else "0"
    elif isinstance(x, int):
        return str(x)
    raise RuntimeError(f"Expected IntImm/IntVar/int/bool, got {type(x)}")


def gen_function_call(func_attrs: Dict[str, Any], indent="  ") -> str:
    y = func_attrs["outputs"][0]

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=y._attrs["name"],
        embed_dim=_to_int_expr(func_attrs["embed_dim"]),
        grid_h=_to_int_expr(func_attrs["grid_h"]),
        grid_w=_to_int_expr(func_attrs["grid_w"]),
        cls_token=_to_int_expr(func_attrs["cls_token"]),
        extra_tokens=_to_int_expr(func_attrs["extra_tokens"]),
        interpolation_scale=str(float(func_attrs["interpolation_scale"])),
        base_size=_to_int_expr(func_attrs["base_size"]),
        indent=indent,
    )


def gen_function(func_attrs, backend_spec):
    func_name = func_attrs["name"]
    backend_spec = CUDASpec()
    out_type = backend_spec.dtype_to_backend_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )
    return SRC_TEMPLATE.render(
        function_name=func_name,
        elem_output_type=out_type,
    )


def gen_function_decl(func_attrs: Dict[str, Any], backend_spec) -> str:
    return FUNC_DECL_TEMPLATE.render(func_name=func_attrs["name"])


@registry.reg("cuda.get_2d_sincos_pos_embed.gen_function")
def cuda_get_2d_sincos_pos_embed_gen_function(func_attrs):
    return gen_function(func_attrs, CUDASpec())


@registry.reg("cuda.get_2d_sincos_pos_embed.func_decl")
def cuda_get_2d_sincos_pos_embed_gen_function_decl(func_attrs: Dict[str, Any]) -> str:
    return gen_function_decl(func_attrs, CUDASpec())


@registry.reg("cuda.get_2d_sincos_pos_embed.func_call")
def cuda_get_2d_sincos_pos_embed_func_call(func_attrs, indent="  "):
    return gen_function_call(func_attrs, indent)
