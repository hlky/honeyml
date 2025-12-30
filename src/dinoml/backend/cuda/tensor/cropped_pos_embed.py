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
    int pos_embed_max_size,
    int base_size,
    float interpolation_scale,
    int patch_size,
    int height,
    int width,
    dinoml::DeviceStream stream
) {
    dinoml::invoke_cropped_pos_embed<{{elem_output_type}}>(
        out,
        embed_dim,
        pos_embed_max_size,
        base_size,
        interpolation_scale,
        patch_size,
        height,
        width,
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
{{indent}}    {{output}},
{{indent}}    {{embed_dim}},
{{indent}}    {{pos_embed_max_size}},
{{indent}}    {{base_size}},
{{indent}}    {{interpolation_scale}},
{{indent}}    {{patch_size}},
{{indent}}    {{height}},
{{indent}}    {{width}},
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


def gen_function_call(func_attrs: Dict[str, Any], indent="  ") -> str:
    y = func_attrs["outputs"][0]
    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=y._attrs["name"],
        embed_dim=_to_int_expr(func_attrs["embed_dim"]),
        pos_embed_max_size=_to_int_expr(func_attrs["pos_embed_max_size"]),
        base_size=_to_int_expr(func_attrs["base_size"]),
        interpolation_scale=str(float(func_attrs["interpolation_scale"])),
        patch_size=_to_int_expr(func_attrs["patch_size"]),
        height=_to_int_expr(func_attrs["height"]),
        width=_to_int_expr(func_attrs["width"]),
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


@registry.reg("cuda.cropped_pos_embed.gen_function")
def cuda_cropped_pos_embed_gen_function(func_attrs):
    return gen_function(func_attrs, CUDASpec())


@registry.reg("cuda.cropped_pos_embed.func_decl")
def cuda_cropped_pos_embed_gen_function_decl(func_attrs: Dict[str, Any]) -> str:
    return gen_function_decl(func_attrs, CUDASpec())


@registry.reg("cuda.cropped_pos_embed.func_call")
def cuda_cropped_pos_embed_func_call(func_attrs, indent="  "):
    return gen_function_call(func_attrs, indent)
