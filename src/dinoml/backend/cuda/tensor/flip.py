from typing import Any, Dict, List

import jinja2

from dinoml.backend import registry
from dinoml.backend.backend_spec import CUDASpec
from dinoml.compiler.base import IntImm, IntVar

SRC_TEMPLATE = jinja2.Template(
    r"""
#include <dinoml/device.h>
#include <ops/flip.h>

void {{function_name}}(
    void* out,
    const void* in,
    int64_t n,
{%- for i in range(rank) %}
    int64_t d{{i}},
{%- endfor %}
    dinoml::DeviceStream stream
) {
    int64_t sizes[{{rank}}] = {
    {%- for i in range(rank) %}
        d{{i}}{{ "," if not loop.last else "" }}
    {%- endfor %}
    };

    int64_t flip_dims[] = {
    {%- for d in flip_dims %}
        {{d}}{{ "," if not loop.last else "" }}
    {%- endfor %}
    };

    invoke_flip<{{elem_output_type}}>(
        out,
        in,
        n,
        sizes,
        flip_dims,
        {{rank}},
        {{flip_dims | length}},
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
    int64_t,
{%- for i in range(rank) %}
    int64_t,
{%- endfor %}
    dinoml::DeviceStream
);
"""
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    r"""
{{indent}}{{func_name}}(
{{indent}}    {{output}},
{{indent}}    {{input}},
{{indent}}    {{numel}},
{%- for d in dim_args %}
{{indent}}    {{d}},
{%- endfor %}
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


def gen_function_call(func_attrs: Dict[str, Any], indent="  ") -> str:
    x = func_attrs["inputs"][0]
    y = func_attrs["outputs"][0]
    shape = x._attrs["shape"]
    numel = gen_int_var_product_str(shape)
    dim_args = [d._attrs["name"] for d in shape]
    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=y._attrs["name"],
        input=x._attrs["name"],
        numel=numel,
        dim_args=dim_args,
        indent=indent,
    )


def gen_function(func_attrs, backend_spec) -> str:
    func_name = func_attrs["name"]
    backend_spec = CUDASpec()
    elem_type = backend_spec.dtype_to_backend_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )

    x = func_attrs["inputs"][0]
    rank = len(x._attrs["shape"])
    flip_dims = func_attrs["flip_dims"]

    if rank > 16:
        raise RuntimeError(
            f"flip: rank {rank} > 16 not supported by current CUDA kernel params"
        )

    return SRC_TEMPLATE.render(
        function_name=func_name,
        elem_output_type=elem_type,
        rank=rank,
        flip_dims=flip_dims,
    )


def gen_function_decl(func_attrs: Dict[str, Any], backend_spec) -> str:
    x = func_attrs["inputs"][0]
    rank = len(x._attrs["shape"])
    return FUNC_DECL_TEMPLATE.render(func_name=func_attrs["name"], rank=rank)


@registry.reg("cuda.flip.gen_function")
def cuda_flip_gen_function(func_attrs):
    return gen_function(func_attrs, CUDASpec())


@registry.reg("cuda.flip.func_decl")
def cuda_flip_gen_function_decl(func_attrs: Dict[str, Any]) -> str:
    return gen_function_decl(func_attrs, CUDASpec())


@registry.reg("cuda.flip.func_call")
def cuda_flip_func_call(func_attrs, indent="  "):
    return gen_function_call(func_attrs, indent)
