from typing import Any, Dict, List

from dinoml.backend import registry
from dinoml.backend.backend_spec import CUDASpec

import jinja2

from dinoml.compiler.base import IntImm, IntVar


SRC_TEMPLATE = jinja2.Template(
    """
#include <dinoml/device.h>
#include <ops/activations.h>

void {{function_name}}(
    void* out,
    const void* in,
    int64_t n,
    int last_dim,
    dinoml::DeviceStream stream
) {
    invoke_gelu_new<{{elem_output_type}}>(
        out,
        in,
        n,
        last_dim,
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
    dinoml::DeviceStream
);
"""
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{output}},
{{indent}}    {{input}},
{{indent}}    {{numel}},
{{indent}}    {{last_dim}},
{{indent}}    stream
{{indent}});
"""
)


def gen_int_var_product_str(
    int_vars: List[IntVar],
) -> str:
    res = []
    for int_var in int_vars:
        if isinstance(int_var, IntImm):
            res.append(str(int_var._attrs["values"][0]))
        elif isinstance(int_var, IntVar):
            res.append(int_var._attrs["name"])
        else:
            raise RuntimeError(
                "A dim must be an IntVar! Current type: {}".format(type(int_var))
            )

    return " * ".join(res) if res else "1"


def gen_function_call(func_attrs: Dict[str, Any], indent="  ") -> str:
    x = func_attrs["inputs"][0]
    y = func_attrs["outputs"][0]

    numel = gen_int_var_product_str(x._attrs["shape"])

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=y._attrs["name"],
        input=x._attrs["name"],
        numel=numel,
        last_dim=x._attrs["shape"][-1]._attrs["name"],
        indent=indent,
    )


def gen_function(func_attrs, template_path):
    func_name = func_attrs["name"]
    backend_spec = CUDASpec()
    input_type = backend_spec.dtype_to_backend_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )
    return SRC_TEMPLATE.render(
        function_name=func_name,
        elem_output_type=input_type,
    )


def gen_function_decl(func_attrs: Dict[str, Any], backend_spec) -> str:
    return FUNC_DECL_TEMPLATE.render(
        func_name=func_attrs["name"],
    )


@registry.reg("cuda.gelu_new.gen_function")
def cuda_gelu_new_gen_function(func_attrs):
    return gen_function(func_attrs, CUDASpec())


@registry.reg("cuda.gelu_new.func_decl")
def cuda_gelu_new_gen_function_decl(func_attrs: Dict[str, Any]) -> str:
    return gen_function_decl(func_attrs, CUDASpec())


@registry.reg("cuda.gelu_new.func_call")
def cuda_gelu_new_func_call(func_attrs, indent="  "):
    return gen_function_call(func_attrs, indent)
