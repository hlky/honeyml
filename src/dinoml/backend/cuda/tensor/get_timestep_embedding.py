from typing import Any, Dict, List

import jinja2

from dinoml.backend import registry
from dinoml.backend.backend_spec import CUDASpec
from dinoml.compiler.base import IntImm, IntVar


SRC_TEMPLATE = jinja2.Template(
    """
#include <dinoml/device.h>
#include <ops/get_timestep_embedding.h>

void {{function_name}}(
    void* out,
    const void* timesteps,
    int64_t n,
    int embedding_dim,
    bool flip_sin_to_cos,
    float downscale_freq_shift,
    float scale,
    int max_period,
    dinoml::DeviceStream stream
) {
    invoke_get_timestep_embedding<{{elem_output_type}}, {{elem_input_type}}>(
        out,
        timesteps,
        n,
        embedding_dim,
        flip_sin_to_cos,
        downscale_freq_shift,
        scale,
        max_period,
        stream
    );
}
"""
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
    void*,
    const void*,
    int64_t,
    int,
    bool,
    float,
    float,
    int,
    dinoml::DeviceStream
);
"""
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{output}},
{{indent}}    {{timesteps}},
{{indent}}    {{n}},
{{indent}}    {{embedding_dim}},
{{indent}}    {{flip_sin_to_cos}},
{{indent}}    {{downscale_freq_shift}},
{{indent}}    {{scale}},
{{indent}}    {{max_period}},
{{indent}}    stream
{{indent}});
"""
)


def gen_int_var_product_str(int_vars: List[IntVar]) -> str:
    res = []
    for int_var in int_vars:
        if isinstance(int_var, IntImm):
            res.append(str(int_var._attrs["values"][0]))
        elif isinstance(int_var, IntVar):
            res.append(int_var._attrs["name"])
        else:
            raise RuntimeError(
                f"A dim must be an IntVar! Current type: {type(int_var)}"
            )
    return " * ".join(res) if res else "1"


def _c_bool(v: bool) -> str:
    return "true" if v else "false"


def _c_float(v: float) -> str:
    # Ensure it's a float literal in C++ (with 'f')
    if isinstance(v, int):
        v = float(v)
    s = repr(float(v))
    if "e" in s or "E" in s or "." in s:
        return f"{s}f"
    return f"{s}.0f"


def gen_function_call(func_attrs: Dict[str, Any], indent="  ") -> str:
    t = func_attrs["inputs"][0]
    y = func_attrs["outputs"][0]

    # timesteps is [N]
    n = gen_int_var_product_str(t._attrs["shape"])

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=y._attrs["name"],
        timesteps=t._attrs["name"],
        n=n,
        embedding_dim=str(func_attrs["embedding_dim"]),
        flip_sin_to_cos=_c_bool(func_attrs["flip_sin_to_cos"]),
        downscale_freq_shift=_c_float(func_attrs["downscale_freq_shift"]),
        scale=_c_float(func_attrs["scale"]),
        max_period=str(func_attrs["max_period"]),
        indent=indent,
    )


def gen_function(func_attrs, backend_spec: CUDASpec):
    func_name = func_attrs["name"]

    out_type = backend_spec.dtype_to_backend_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )
    in_type = backend_spec.dtype_to_backend_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )

    return SRC_TEMPLATE.render(
        function_name=func_name,
        elem_output_type=out_type,
        elem_input_type=in_type,
    )


def gen_function_decl(func_attrs: Dict[str, Any], backend_spec) -> str:
    return FUNC_DECL_TEMPLATE.render(func_name=func_attrs["name"])


@registry.reg("cuda.get_timestep_embedding.gen_function")
def cuda_get_timestep_embedding_gen_function(func_attrs):
    return gen_function(func_attrs, CUDASpec())


@registry.reg("cuda.get_timestep_embedding.func_decl")
def cuda_get_timestep_embedding_gen_function_decl(func_attrs: Dict[str, Any]) -> str:
    return gen_function_decl(func_attrs, CUDASpec())


@registry.reg("cuda.get_timestep_embedding.func_call")
def cuda_get_timestep_embedding_func_call(func_attrs, indent="  "):
    return gen_function_call(func_attrs, indent)
