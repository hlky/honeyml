from typing import Any, Dict, List

import jinja2

from dinoml.backend import registry
from dinoml.backend.backend_spec import CUDASpec
from dinoml.compiler.base import IntImm, IntVar


SRC_TEMPLATE = jinja2.Template(
    r"""
#include <dinoml/device.h>
#include <ops/rotary_positional_embeddings_1d.h>

void {{function_name}}(
    void* out0,
    void* out1,
    const void* pos_ptr,
    int pos_is_tensor,
    int64_t S,
    int dim,
    float theta,
    int use_real,
    float linear_factor,
    float ntk_factor,
    int repeat_interleave_real,
    dinoml::DeviceStream stream
) {
    dinoml::invoke_get_1d_rotary_pos_embed<{{elem_output_type}}>(
        out0,
        out1,
        pos_ptr,
        pos_is_tensor,
        S,
        dim,
        theta,
        use_real,
        linear_factor,
        ntk_factor,
        repeat_interleave_real,
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
    const void*,
    int,
    int64_t,
    int,
    float,
    int,
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
{{indent}}    {{out0}},
{{indent}}    {{out1}},
{{indent}}    {{pos_ptr}},
{{indent}}    {{pos_is_tensor}},
{{indent}}    {{S}},
{{indent}}    {{dim}},
{{indent}}    {{theta}},
{{indent}}    {{use_real}},
{{indent}}    {{linear_factor}},
{{indent}}    {{ntk_factor}},
{{indent}}    {{repeat_interleave_real}},
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
    if isinstance(x, int):
        return str(x)
    raise RuntimeError(f"Expected IntImm/IntVar/int/bool, got {type(x)}")


def _c_float(v: float) -> str:
    s = repr(float(v))
    if "e" in s or "E" in s or "." in s:
        return f"{s}f"
    return f"{s}.0f"


def gen_int_var_product_str(int_vars: List[IntVar]) -> str:
    res = []
    for v in int_vars:
        if isinstance(v, IntImm):
            res.append(str(v._attrs["values"][0]))
        elif isinstance(v, IntVar):
            res.append(v._attrs["name"])
        else:
            raise RuntimeError(f"A dim must be IntVar/IntImm. Got {type(v)}")
    return " * ".join(res) if res else "1"


def gen_function_call(func_attrs: Dict[str, Any], indent="  ") -> str:
    out0 = func_attrs["outputs"][0]
    out1 = func_attrs["outputs"][1]

    if func_attrs["pos_is_tensor"]:
        pos = func_attrs["inputs"][0]
        pos_ptr = pos._attrs["name"]
        S = gen_int_var_product_str(pos._attrs["shape"])  # pos is [S]
        pos_is_tensor = "1"
    else:
        pos_ptr = "nullptr"
        S = str(int(func_attrs["pos_int"]))
        pos_is_tensor = "0"

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        out0=out0._attrs["name"],
        out1=out1._attrs["name"],
        pos_ptr=pos_ptr,
        pos_is_tensor=pos_is_tensor,
        S=S,
        dim=_to_int_expr(func_attrs["dim"]),
        theta=_c_float(func_attrs["theta"]),
        use_real=_to_int_expr(func_attrs["use_real"]),
        linear_factor=_c_float(func_attrs["linear_factor"]),
        ntk_factor=_c_float(func_attrs["ntk_factor"]),
        repeat_interleave_real=_to_int_expr(func_attrs["repeat_interleave_real"]),
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


@registry.reg("cuda.get_1d_rotary_pos_embed.gen_function")
def cuda_get_1d_rotary_pos_embed_gen_function(func_attrs):
    return gen_function(func_attrs, CUDASpec())


@registry.reg("cuda.get_1d_rotary_pos_embed.func_decl")
def cuda_get_1d_rotary_pos_embed_gen_function_decl(func_attrs: Dict[str, Any]) -> str:
    return gen_function_decl(func_attrs, CUDASpec())


@registry.reg("cuda.get_1d_rotary_pos_embed.func_call")
def cuda_get_1d_rotary_pos_embed_func_call(func_attrs, indent="  "):
    return gen_function_call(func_attrs, indent)
