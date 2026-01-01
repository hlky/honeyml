import jinja2
from dinoml.backend import registry
from dinoml.backend.backend_spec import CUDASpec
from dinoml.compiler.base import IntImm, IntVar

SRC_TEMPLATE = jinja2.Template(
    r"""
#include <dinoml/device.h>
#include <ops/sinusoidal_positional_embedding.h>

void {{function_name}}(
    void* out,
    const void* x,
    int batch,
    int seq_len,
    int embed_dim,
    dinoml::DeviceStream stream
) {
    dinoml::invoke_sinusoidal_positional_embedding<{{elem_type}}>(
        out, x, batch, seq_len, embed_dim, stream
    );
}
"""
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    r"""
void {{func_name}}(
    void*,
    const void*,
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
{{indent}}    {{out}},
{{indent}}    {{x}},
{{indent}}    {{batch}},
{{indent}}    {{seq_len}},
{{indent}}    {{embed_dim}},
{{indent}}    stream
{{indent}});
"""
)


def gen_function(func_attrs, backend_spec):
    return SRC_TEMPLATE.render(
        function_name=func_attrs["name"],
        elem_type=backend_spec.dtype_to_backend_type(
            func_attrs["outputs"][0]._attrs["dtype"]
        ),
    )


def gen_function_decl(func_attrs, backend_spec):
    return FUNC_DECL_TEMPLATE.render(func_name=func_attrs["name"])


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


def gen_function_call(func_attrs, indent="  "):
    x = func_attrs["inputs"][0]
    y = func_attrs["outputs"][0]
    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        out=y._attrs["name"],
        x=x._attrs["name"],
        batch=_to_int_expr(x._attrs["shape"][0]),
        seq_len=_to_int_expr(x._attrs["shape"][1]),
        embed_dim=_to_int_expr(x._attrs["shape"][2]),
        indent=indent,
    )


@registry.reg("cuda.sinusoidal_positional_embedding.gen_function")
def gen_fn(attrs):
    return gen_function(attrs, CUDASpec())


@registry.reg("cuda.sinusoidal_positional_embedding.func_decl")
def gen_decl(attrs):
    return gen_function_decl(attrs, CUDASpec())


@registry.reg("cuda.sinusoidal_positional_embedding.func_call")
def gen_call(attrs, indent="  "):
    return gen_function_call(attrs, indent)
