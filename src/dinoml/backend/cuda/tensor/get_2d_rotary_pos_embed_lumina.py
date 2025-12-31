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
    void* out_real,
    void* out_imag,
    int embed_dim,
    int len_h,
    int len_w,
    float linear_factor,
    float ntk_factor,
    dinoml::DeviceStream stream
) {
    dinoml::invoke_get_2d_rotary_pos_embed_lumina<{{elem_output_type}}>(
        out_real,
        out_imag,
        embed_dim,
        len_h,
        len_w,
        linear_factor,
        ntk_factor,
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
    int,
    int,
    float,
    float,
    dinoml::DeviceStream
);
"""
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    r"""
{{indent}}{{func_name}}(
{{indent}}    {{out_real}},
{{indent}}    {{out_imag}},
{{indent}}    {{embed_dim}},
{{indent}}    {{len_h}},
{{indent}}    {{len_w}},
{{indent}}    {{linear_factor}},
{{indent}}    {{ntk_factor}},
{{indent}}    stream
{{indent}});
"""
)


def _to_float_lit(x: float) -> str:
    s = repr(float(x))
    if "e" in s or "E" in s or "." in s:
        return f"{s}f"
    return f"{s}.0f"


def _to_int_expr(x) -> str:
    if isinstance(x, IntImm):
        return str(x.value())
    if isinstance(x, IntVar):
        return x._attrs["name"]
    if isinstance(x, int):
        return str(x)
    raise RuntimeError(f"Expected IntImm/IntVar/int, got {type(x)}")


def gen_function_call(func_attrs: Dict[str, Any], indent="  ") -> str:
    out_real = func_attrs["outputs"][0]
    out_imag = func_attrs["outputs"][1]

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        out_real=out_real._attrs["name"],
        out_imag=out_imag._attrs["name"],
        embed_dim=_to_int_expr(func_attrs["embed_dim"]),
        len_h=_to_int_expr(func_attrs["len_h"]),
        len_w=_to_int_expr(func_attrs["len_w"]),
        linear_factor=_to_float_lit(func_attrs["linear_factor"]),
        ntk_factor=_to_float_lit(func_attrs["ntk_factor"]),
        indent=indent,
    )


def gen_function(func_attrs, backend_spec):
    out_type = backend_spec.dtype_to_backend_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )
    return SRC_TEMPLATE.render(
        function_name=func_attrs["name"],
        elem_output_type=out_type,
    )


def gen_function_decl(func_attrs: Dict[str, Any], backend_spec) -> str:
    return FUNC_DECL_TEMPLATE.render(func_name=func_attrs["name"])


@registry.reg("cuda.get_2d_rotary_pos_embed_lumina.gen_function")
def cuda_get_2d_rotary_pos_embed_lumina_gen_function(func_attrs):
    return gen_function(func_attrs, CUDASpec())


@registry.reg("cuda.get_2d_rotary_pos_embed_lumina.func_decl")
def cuda_get_2d_rotary_pos_embed_lumina_gen_function_decl(
    func_attrs: Dict[str, Any],
) -> str:
    return gen_function_decl(func_attrs, CUDASpec())


@registry.reg("cuda.get_2d_rotary_pos_embed_lumina.func_call")
def cuda_get_2d_rotary_pos_embed_lumina_func_call(func_attrs, indent="  "):
    return gen_function_call(func_attrs, indent)
