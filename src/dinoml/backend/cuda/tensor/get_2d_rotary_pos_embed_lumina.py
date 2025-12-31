from typing import Dict
import jinja2
from dinoml.backend import registry
from dinoml.backend.backend_spec import CUDASpec
from dinoml.compiler.base import IntImm, IntVar


SRC_TEMPLATE = jinja2.Template(
    r"""
#include <dinoml/device.h>
#include <ops/rotary_positional_embeddings.h>

void {{function_name}}(
    void* out,
    int embed_dim,
    int grid_h,
    int grid_w,
    float linear_factor,
    float ntk_factor,
    dinoml::DeviceStream stream
) {
    dinoml::invoke_get_2d_rotary_pos_embed_lumina<{{elem_type}}>(
        out,
        embed_dim,
        grid_h,
        grid_w,
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
{{indent}}    {{output}},
{{indent}}    {{embed_dim}},
{{indent}}    {{grid_h}},
{{indent}}    {{grid_w}},
{{indent}}    {{linear_factor}},
{{indent}}    {{ntk_factor}},
{{indent}}    stream
{{indent}});
"""
)


def gen_function(func_attrs, backend_spec):
    return SRC_TEMPLATE.render(
        function_name=func_attrs["name"],
        elem_type=backend_spec.dtype_to_backend_type(
            func_attrs["outputs"][0]._attrs["dtype"]
        ),
    )


def gen_function_decl(func_attrs, backend_spec):
    return FUNC_DECL_TEMPLATE.render(func_name=func_attrs["name"])


def _to_int_expr(x) -> str:
    if isinstance(x, IntImm):
        return str(x.value())
    if isinstance(x, IntVar):
        return x._attrs["name"]
    if isinstance(x, int):
        return str(x)
    raise RuntimeError(f"Expected IntImm/IntVar/int, got {type(x)}")


def gen_function_call(func_attrs, indent="  "):
    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=func_attrs["outputs"][0]._attrs["name"],
        embed_dim=_to_int_expr(func_attrs["embed_dim"]),
        grid_h=_to_int_expr(func_attrs["grid_h"]),
        grid_w=_to_int_expr(func_attrs["grid_w"]),
        linear_factor=func_attrs["linear_factor"],
        ntk_factor=func_attrs["ntk_factor"],
        indent=indent,
    )


@registry.reg("cuda.get_2d_rotary_pos_embed_lumina.gen_function")
def _(func_attrs):
    return gen_function(func_attrs, CUDASpec())


@registry.reg("cuda.get_2d_rotary_pos_embed_lumina.func_decl")
def _(func_attrs):
    return gen_function_decl(func_attrs, CUDASpec())


@registry.reg("cuda.get_2d_rotary_pos_embed_lumina.func_call")
def _(func_attrs, indent="  "):
    return gen_function_call(func_attrs, indent)
