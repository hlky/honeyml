from typing import Any, Dict, List

import jinja2

from dinoml.backend import registry
from dinoml.backend.backend_spec import CUDASpec
from dinoml.compiler.base import IntImm, IntVar


SRC_TEMPLATE = jinja2.Template(
    r"""
#include <dinoml/device.h>
#include <ops/gaussian_fourier_projection.h>

void {{function_name}}(
    void* out,
    const void* x,
    const void* weight,
    int64_t n,
    int embedding_size,
    int do_log,
    int flip_sin_to_cos,
    dinoml::DeviceStream stream
) {
    dinoml::invoke_gaussian_fourier_projection<{{elem_type}}>(
        out,
        x,
        weight,
        n,
        embedding_size,
        do_log,
        flip_sin_to_cos,
        stream
    );
}
"""
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    r"""
void {{func_name}}(
    void*,
    const void*,
    const void*,
    int64_t,
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
{{indent}}    {{x}},
{{indent}}    {{weight}},
{{indent}}    {{n}},
{{indent}}    {{embedding_size}},
{{indent}}    {{do_log}},
{{indent}}    {{flip_sin_to_cos}},
{{indent}}    stream
{{indent}});
"""
)


def gen_int_var_product_str(int_vars: List[IntVar]) -> str:
    res = []
    for v in int_vars:
        if isinstance(v, IntImm):
            res.append(str(v._attrs["values"][0]))
        elif isinstance(v, IntVar):
            res.append(v._attrs["name"])
        else:
            raise RuntimeError(f"Expected IntImm/IntVar, got {type(v)}")
    return " * ".join(res) if res else "1"


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


def gen_function_call(func_attrs: Dict[str, Any], indent="  ") -> str:
    x = func_attrs["inputs"][0]
    w = func_attrs["inputs"][1]
    y = func_attrs["outputs"][0]

    # x is expected to be [N] or [N,1]? We treat as [N] and use product of shape dims
    n = gen_int_var_product_str(x._attrs["shape"])

    # embedding_size comes from weight last dim (static or symbolic)
    # store normalized in operator as int/intvar
    emb = func_attrs["embedding_size"]
    if isinstance(emb, (IntImm, IntVar)):
        emb_expr = _to_int_expr(emb)
    else:
        emb_expr = str(int(emb))

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=y._attrs["name"],
        x=x._attrs["name"],
        weight=w._attrs["name"],
        n=n,
        embedding_size=emb_expr,
        do_log=_to_int_expr(func_attrs["log"]),
        flip_sin_to_cos=_to_int_expr(func_attrs["flip_sin_to_cos"]),
        indent=indent,
    )


def gen_function(func_attrs, backend_spec):
    backend_spec = CUDASpec()
    elem_type = backend_spec.dtype_to_backend_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )
    return SRC_TEMPLATE.render(
        function_name=func_attrs["name"],
        elem_type=elem_type,
    )


def gen_function_decl(func_attrs: Dict[str, Any], backend_spec) -> str:
    return FUNC_DECL_TEMPLATE.render(func_name=func_attrs["name"])


@registry.reg("cuda.gaussian_fourier_projection.gen_function")
def cuda_gaussian_fourier_projection_gen_function(func_attrs):
    return gen_function(func_attrs, CUDASpec())


@registry.reg("cuda.gaussian_fourier_projection.func_decl")
def cuda_gaussian_fourier_projection_gen_function_decl(
    func_attrs: Dict[str, Any],
) -> str:
    return gen_function_decl(func_attrs, CUDASpec())


@registry.reg("cuda.gaussian_fourier_projection.func_call")
def cuda_gaussian_fourier_projection_func_call(func_attrs, indent="  "):
    return gen_function_call(func_attrs, indent)
