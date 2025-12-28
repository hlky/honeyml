from typing import List
import jinja2

from dinoml.backend import registry
from dinoml.backend.backend_spec import CUDASpec
from dinoml.compiler.base import IntImm, IntVar

SRC_TEMPLATE = jinja2.Template(
    """
#include <dinoml/device.h>
#include <ops/randn.h>

void {{function_name}}(
    void* out_ptr,
    int64_t n,
    float mean,
    float std,
    uint64_t seed,
    uint64_t offset_groups,
    dinoml::DeviceStream stream
) {
    invoke_randn<{{elem_output_type}}>(
        out_ptr,
        n,
        mean,
        std,
        seed,
        offset_groups,
        stream
    );
}
"""
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
    void*,
    int64_t,
    float,
    float,
    uint64_t,
    uint64_t,
    dinoml::DeviceStream
);
"""
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{
    {{indent}}uint64_t {{func_name}}_n_elements = {{n}};
    {{indent}}uint64_t {{func_name}}_seed = {% if seed is not none %}{{ seed }}{% else %}dinoml::make_seed(){% endif %};
    {{indent}}uint64_t {{func_name}}_counter = {% if offset_groups is not none %}{{ offset_groups }}{% else %}global_counter_{% endif %};
    {{indent}}  auto [{{func_name}}_counter_offset, {{func_name}}_grid, {{func_name}}_block] =
    {{indent}}      calc_execution_policy((int64_t){{func_name}}_n_elements, /*unroll_factor=*/4);
    {{indent}}{{func_name}}(
    {{indent}}    {{out_ptr}},
    {{indent}}    {{func_name}}_n_elements,
    {{indent}}    {{mean}},
    {{indent}}    {{std}},
    {{indent}}    {{func_name}}_seed,
    {{indent}}    {{func_name}}_counter,
    {{indent}}    stream
    {{indent}});
    {% if offset_groups is none %}
    {{indent}}uint64_t {{func_name}}_groups = ({{func_name}}_n_elements + 3) / 4;
    {{indent}}global_counter_ += {{func_name}}_counter_offset;
    {% endif %}
{{indent}}}
"""
)


def randn_gen_function_decl(func_attrs, backend_spec):
    """Function declaration generation

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        It describes the operation attributes
    backend_spec : custom class
        It specifies the corresponding backend dtypes of pytorch dtypes for many operations

    Returns
    -------
    str
        Rendered function declaration stmt
    """
    return FUNC_DECL_TEMPLATE.render(
        func_name=func_attrs["name"],
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


def randn_gen_function_call(func_attrs, backend_spec, indent="  "):
    """Function call generation

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        It describes the operation attributes
    indent : str, optional
        Indent for template, by default "  "

    Returns
    -------
    str
        Rendered function call
    """
    x = func_attrs["outputs"][0]
    x_shape = x._attrs["shape"]
    n = gen_int_var_product_str(x_shape)

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        out_ptr=x._attrs["name"],
        n=n,
        mean=func_attrs["mean"],
        std=func_attrs["std"],
        seed=func_attrs["seed"],
        offset_groups=func_attrs["offset_groups"],
        indent=indent,
    )


@registry.reg("cuda.randn.func_decl")
def gen_function_decl(func_attrs):
    return randn_gen_function_decl(func_attrs, backend_spec=CUDASpec())


@registry.reg("cuda.randn.func_call")
def gen_function_call(func_attrs, indent="    "):
    return randn_gen_function_call(func_attrs, backend_spec=CUDASpec(), indent=indent)


@registry.reg("cuda.randn.gen_function")
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
