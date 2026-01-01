from typing import Any, Dict

import jinja2

from dinoml.backend import registry
from dinoml.backend.backend_spec import CUDASpec
from dinoml.compiler.base import IntImm, IntVar


SRC_TEMPLATE = jinja2.Template(
    r"""
#include <dinoml/device.h>
#include <ops/rotary_positional_embeddings_allegro.h>

void {{function_name}}(
    void* freqs_t_cos,
    void* freqs_t_sin,
    void* freqs_h_cos,
    void* freqs_h_sin,
    void* freqs_w_cos,
    void* freqs_w_sin,
    void* grid_t,
    void* grid_h,
    void* grid_w,
    int height,
    int width,
    int num_frames,
    int vae_scale_factor_spatial,
    int patch_size,
    float interpolation_scale_h,
    float interpolation_scale_t,
    float interpolation_scale_w,
    int attention_head_dim,
    dinoml::DeviceStream stream
) {
    dinoml::invoke_get_3d_rotary_pos_embed_allegro<{{elem_type}}>(
        freqs_t_cos,
        freqs_t_sin,
        freqs_h_cos,
        freqs_h_sin,
        freqs_w_cos,
        freqs_w_sin,
        grid_t,
        grid_h,
        grid_w,
        height,
        width,
        num_frames,
        vae_scale_factor_spatial,
        patch_size,
        interpolation_scale_h,
        interpolation_scale_t,
        interpolation_scale_w,
        attention_head_dim,
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
    void*,
    void*,
    void*,
    void*,
    void*,
    void*,
    void*,
    int,
    int,
    int,
    int,
    int,
    float,
    float,
    float,
    int,
    dinoml::DeviceStream
);
"""
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    r"""
{{indent}}{{func_name}}(
{{indent}}    {{freqs_t_cos}},
{{indent}}    {{freqs_t_sin}},
{{indent}}    {{freqs_h_cos}},
{{indent}}    {{freqs_h_sin}},
{{indent}}    {{freqs_w_cos}},
{{indent}}    {{freqs_w_sin}},
{{indent}}    {{grid_t}},
{{indent}}    {{grid_h}},
{{indent}}    {{grid_w}},
{{indent}}    {{height}},
{{indent}}    {{width}},
{{indent}}    {{num_frames}},
{{indent}}    {{vae_scale_factor_spatial}},
{{indent}}    {{patch_size}},
{{indent}}    {{interpolation_scale_h}},
{{indent}}    {{interpolation_scale_t}},
{{indent}}    {{interpolation_scale_w}},
{{indent}}    {{attention_head_dim}},
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
    v = float(x)
    s = repr(v)
    if "e" in s or "E" in s or "." in s:
        return f"{s}f"
    return f"{s}.0f"


def gen_function_call(func_attrs: Dict[str, Any], indent="  ") -> str:
    outs = func_attrs["outputs"]
    # outputs order:
    # 0:t_cos 1:t_sin 2:h_cos 3:h_sin 4:w_cos 5:w_sin 6:grid_t 7:grid_h 8:grid_w
    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        freqs_t_cos=outs[0]._attrs["name"],
        freqs_t_sin=outs[1]._attrs["name"],
        freqs_h_cos=outs[2]._attrs["name"],
        freqs_h_sin=outs[3]._attrs["name"],
        freqs_w_cos=outs[4]._attrs["name"],
        freqs_w_sin=outs[5]._attrs["name"],
        grid_t=outs[6]._attrs["name"],
        grid_h=outs[7]._attrs["name"],
        grid_w=outs[8]._attrs["name"],
        height=_to_int_expr(func_attrs["height"]),
        width=_to_int_expr(func_attrs["width"]),
        num_frames=_to_int_expr(func_attrs["num_frames"]),
        vae_scale_factor_spatial=_to_int_expr(func_attrs["vae_scale_factor_spatial"]),
        patch_size=_to_int_expr(func_attrs["patch_size"]),
        interpolation_scale_h=_to_float_lit(func_attrs["interpolation_scale_h"]),
        interpolation_scale_t=_to_float_lit(func_attrs["interpolation_scale_t"]),
        interpolation_scale_w=_to_float_lit(func_attrs["interpolation_scale_w"]),
        attention_head_dim=_to_int_expr(func_attrs["attention_head_dim"]),
        indent=indent,
    )


def gen_function(func_attrs, backend_spec):
    backend_spec = CUDASpec()
    # freqs outputs dtype determines elem_type; grids are int64 but generated in same function
    elem_type = backend_spec.dtype_to_backend_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )
    return SRC_TEMPLATE.render(function_name=func_attrs["name"], elem_type=elem_type)


def gen_function_decl(func_attrs: Dict[str, Any], backend_spec) -> str:
    return FUNC_DECL_TEMPLATE.render(func_name=func_attrs["name"])


@registry.reg("cuda.get_3d_rotary_pos_embed_allegro.gen_function")
def cuda_get_3d_rotary_pos_embed_allegro_gen_function(func_attrs):
    return gen_function(func_attrs, CUDASpec())


@registry.reg("cuda.get_3d_rotary_pos_embed_allegro.func_decl")
def cuda_get_3d_rotary_pos_embed_allegro_gen_function_decl(
    func_attrs: Dict[str, Any],
) -> str:
    return gen_function_decl(func_attrs, CUDASpec())


@registry.reg("cuda.get_3d_rotary_pos_embed_allegro.func_call")
def cuda_get_3d_rotary_pos_embed_allegro_func_call(func_attrs, indent="  "):
    return gen_function_call(func_attrs, indent)
