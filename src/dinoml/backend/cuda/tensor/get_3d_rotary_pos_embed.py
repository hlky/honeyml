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
    int grid_size_h,
    int grid_size_w,
    int temporal_size,
    float theta,
    int grid_type,
    int max_h,
    int max_w,
    dinoml::DeviceStream stream
) {
    dinoml::invoke_get_3d_rotary_pos_embed<{{elem_output_type}}>(
        out_cos,
        out_sin,
        embed_dim,
        crop_start_h,
        crop_start_w,
        crop_stop_h,
        crop_stop_w,
        grid_size_h,
        grid_size_w,
        temporal_size,
        theta,
        grid_type,
        max_h,
        max_w,
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
    int,
    float,
    int,
    int,
    int,
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
{{indent}}    {{grid_size_h}},
{{indent}}    {{grid_size_w}},
{{indent}}    {{temporal_size}},
{{indent}}    {{theta}},
{{indent}}    {{grid_type}},
{{indent}}    {{max_h}},
{{indent}}    {{max_w}},
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


def _to_float_lit(x: float) -> str:
    # ensure C++ float literal
    v = float(x)
    s = repr(v)
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
        crop_start_h=_to_float_lit(func_attrs["crop_start_h"]),
        crop_start_w=_to_float_lit(func_attrs["crop_start_w"]),
        crop_stop_h=_to_float_lit(func_attrs["crop_stop_h"]),
        crop_stop_w=_to_float_lit(func_attrs["crop_stop_w"]),
        grid_size_h=_to_int_expr(func_attrs["grid_size_h"]),
        grid_size_w=_to_int_expr(func_attrs["grid_size_w"]),
        temporal_size=_to_int_expr(func_attrs["temporal_size"]),
        theta=_to_float_lit(func_attrs["theta"]),
        grid_type=_to_int_expr(func_attrs["grid_type"]),
        max_h=_to_int_expr(func_attrs["max_h"]),
        max_w=_to_int_expr(func_attrs["max_w"]),
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


@registry.reg("cuda.get_3d_rotary_pos_embed.gen_function")
def cuda_get_3d_rotary_pos_embed_gen_function(func_attrs):
    return gen_function(func_attrs, CUDASpec())


@registry.reg("cuda.get_3d_rotary_pos_embed.func_decl")
def cuda_get_3d_rotary_pos_embed_gen_function_decl(func_attrs: Dict[str, Any]) -> str:
    return gen_function_decl(func_attrs, CUDASpec())


@registry.reg("cuda.get_3d_rotary_pos_embed.func_call")
def cuda_get_3d_rotary_pos_embed_func_call(func_attrs, indent="  "):
    return gen_function_call(func_attrs, indent)
