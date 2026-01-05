from dinoml.backend import registry
from dinoml.backend.backend_spec import CUDASpec
import jinja2

from dinoml.compiler.base import IntImm, IntVar

SRC_TEMPLATE = jinja2.Template(
    """
#include <dinoml/device.h>
#include <ops/kdownsample2d_weight.h>

void {{func_name}}(
    void* out,
    int channels,
    dinoml::DeviceStream stream
) {
    invoke_kdownsample2d_weight<{{dtype}}>(out, channels, stream);
}
"""
)

FUNC_DECL = jinja2.Template(
    """
void {{func_name}}(void*, int, dinoml::DeviceStream);
"""
)


FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}   {{output}}, {{channels}}, stream /* default stream */
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


def gen_function_call(func_attrs, backend_spec, indent="  "):
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
    channels = _to_int_expr(func_attrs["channels"])
    y = func_attrs["outputs"][0]
    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=y._attrs["name"],
        channels=channels,
        indent=indent,
    )


def gen_function(func_attrs, backend_spec):
    dtype = backend_spec.dtype_to_backend_type(func_attrs["dtype"])
    return SRC_TEMPLATE.render(
        func_name=func_attrs["name"],
        dtype=dtype,
    )


@registry.reg("cuda.kdownsample2d_weight.gen_function")
def cuda_kdownsample2d_weight_gen_function(attrs, indent="  "):
    return gen_function(attrs, CUDASpec())


@registry.reg("cuda.kdownsample2d_weight.func_decl")
def cuda_kdownsample2d_weight_gen_decl(attrs):
    return FUNC_DECL.render(func_name=attrs["name"])


@registry.reg("cuda.kdownsample2d_weight.func_call")
def cuda_kdownsample2d_weight_gen_function_call(attrs, indent="  "):
    return gen_function_call(attrs, CUDASpec())
