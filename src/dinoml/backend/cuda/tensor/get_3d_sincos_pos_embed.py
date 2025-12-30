from typing import Any, Dict, Tuple, Union

from dinoml.backend import registry
from dinoml.backend.backend_spec import CUDASpec

import jinja2

from dinoml.compiler.base import IntImm, IntVar


SRC_TEMPLATE = jinja2.Template(
    r"""
#include <dinoml/device.h>
#include <ops/positional_embeddings.h>

void {{function_name}}(
    void* out,
    int embed_dim,
    int spatial_w,
    int spatial_h,
    int temporal_size,
    float spatial_interpolation_scale,
    float temporal_interpolation_scale,
    dinoml::DeviceStream stream
) {
    dinoml::invoke_sincos_pos_embed_3d<{{elem_output_type}}>(
        out,
        embed_dim,
        spatial_w,
        spatial_h,
        temporal_size,
        spatial_interpolation_scale,
        temporal_interpolation_scale,
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
{{indent}}    {{spatial_w}},
{{indent}}    {{spatial_h}},
{{indent}}    {{temporal_size}},
{{indent}}    {{spatial_interpolation_scale}},
{{indent}}    {{temporal_interpolation_scale}},
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
    raise RuntimeError(f"Expected IntImm or IntVar, got {type(x)}")


def gen_function_call(func_attrs: Dict[str, Any], indent="  ") -> str:
    y = func_attrs["outputs"][0]

    # stored attrs are normalized in the Operator
    embed_dim = _to_int_expr(func_attrs["embed_dim"])
    spatial_w = _to_int_expr(func_attrs["spatial_w"])
    spatial_h = _to_int_expr(func_attrs["spatial_h"])
    temporal_size = _to_int_expr(func_attrs["temporal_size"])

    spatial_interpolation_scale = str(float(func_attrs["spatial_interpolation_scale"]))
    temporal_interpolation_scale = str(
        float(func_attrs["temporal_interpolation_scale"])
    )

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=y._attrs["name"],
        embed_dim=embed_dim,
        spatial_w=spatial_w,
        spatial_h=spatial_h,
        temporal_size=temporal_size,
        spatial_interpolation_scale=spatial_interpolation_scale,
        temporal_interpolation_scale=temporal_interpolation_scale,
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
    return FUNC_DECL_TEMPLATE.render(
        func_name=func_attrs["name"],
    )


@registry.reg("cuda.get_3d_sincos_pos_embed.gen_function")
def cuda_get_3d_sincos_pos_embed_gen_function(func_attrs):
    return gen_function(func_attrs, CUDASpec())


@registry.reg("cuda.get_3d_sincos_pos_embed.func_decl")
def cuda_get_3d_sincos_pos_embed_gen_function_decl(func_attrs: Dict[str, Any]) -> str:
    return gen_function_decl(func_attrs, CUDASpec())


@registry.reg("cuda.get_3d_sincos_pos_embed.func_call")
def cuda_get_3d_sincos_pos_embed_func_call(func_attrs, indent="  "):
    return gen_function_call(func_attrs, indent)
