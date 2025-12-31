from typing import Any, Dict

import jinja2

from dinoml.backend import registry
from dinoml.backend.backend_spec import CUDASpec
from dinoml.compiler.base import IntImm, IntVar


SRC_TEMPLATE = jinja2.Template(
    r"""
#include <dinoml/device.h>
#include <ops/rotary_positional_embeddings.h>

void {{function_name}}(
    void* out_cos,
    void* out_sin,
    int embed_dim,
    float crop_start_h,
    float crop_start_w,
    float crop_stop_h,
    float crop_stop_w,
    int grid_h,
    int grid_w,
    float theta,
    dinoml::DeviceStream stream
) {
    dinoml::invoke_get_2d_rotary_pos_embed<{{elem_output_type}}>(
        out_cos,
        out_sin,
        embed_dim,
        crop_start_h,
        crop_start_w,
        crop_stop_h,
        crop_stop_w,
        grid_h,
        grid_w,
        theta,
        stream
    );
}
"""
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    r"""
void {{func_name}}(
    void*,
    void*,
    int,
    float,
    float,
    float,
    float,
    int,
    int,
    float,
    dinoml::DeviceStream
);
"""
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    r"""
{{indent}}{{func_name}}(
{{indent}}    {{out_cos}},
{{indent}}    {{out_sin}},
{{indent}}    {{embed_dim}},
{{indent}}    {{crop_start_h}},
{{indent}}    {{crop_start_w}},
{{indent}}    {{crop_stop_h}},
{{indent}}    {{crop_stop_w}},
{{indent}}    {{grid_h}},
{{indent}}    {{grid_w}},
{{indent}}    {{theta}},
{{indent}}    stream
{{indent}});
"""
)


def _to_int_expr(x) -> str:
    if isinstance(x, IntImm):
        return str(x.value())
    if isinstance(x, IntVar):
        return x._attrs["name"]
    if isinstance(x, int):
        return str(x)
    raise RuntimeError(f"Expected IntImm/IntVar/int, got {type(x)}")


def _c_float(v: float) -> str:
    s = repr(float(v))
    if "e" in s or "E" in s or "." in s:
        return f"{s}f"
    return f"{s}.0f"


def gen_function_call(func_attrs: Dict[str, Any], indent="  ") -> str:
    out_cos = func_attrs["outputs"][0]
    out_sin = func_attrs["outputs"][1]

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        out_cos=out_cos._attrs["name"],
        out_sin=out_sin._attrs["name"],
        embed_dim=_to_int_expr(func_attrs["embed_dim"]),
        crop_start_h=_c_float(func_attrs["crop_start_h"]),
        crop_start_w=_c_float(func_attrs["crop_start_w"]),
        crop_stop_h=_c_float(func_attrs["crop_stop_h"]),
        crop_stop_w=_c_float(func_attrs["crop_stop_w"]),
        grid_h=_to_int_expr(func_attrs["grid_h"]),
        grid_w=_to_int_expr(func_attrs["grid_w"]),
        theta=_c_float(func_attrs["theta"]),
        indent=indent,
    )


def gen_function(func_attrs, backend_spec):
    backend_spec = CUDASpec()
    out_type = backend_spec.dtype_to_backend_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )
    return SRC_TEMPLATE.render(
        function_name=func_attrs["name"], elem_output_type=out_type
    )


def gen_function_decl(func_attrs: Dict[str, Any], backend_spec) -> str:
    return FUNC_DECL_TEMPLATE.render(func_name=func_attrs["name"])


@registry.reg("cuda.get_2d_rotary_pos_embed.gen_function")
def cuda_get_2d_rotary_pos_embed_gen_function(func_attrs):
    return gen_function(func_attrs, CUDASpec())


@registry.reg("cuda.get_2d_rotary_pos_embed.func_decl")
def cuda_get_2d_rotary_pos_embed_gen_function_decl(func_attrs: Dict[str, Any]) -> str:
    return gen_function_decl(func_attrs, CUDASpec())


@registry.reg("cuda.get_2d_rotary_pos_embed.func_call")
def cuda_get_2d_rotary_pos_embed_func_call(func_attrs, indent="  "):
    return gen_function_call(func_attrs, indent)
