from typing import Any, Dict

import jinja2

from dinoml.backend import registry
from dinoml.backend.backend_spec import CUDASpec
from dinoml.compiler.base import IntImm, IntVar


SRC_TEMPLATE = jinja2.Template(
    r"""
#include <dinoml/device.h>
#include <ops/fourier_embeds_from_boundingbox.h>

void {{function_name}}(
    void* out,
    const void* box,
    int embed_dim,
    int batch_size,
    int num_boxes,
    dinoml::DeviceStream stream
) {
    dinoml::invoke_get_fourier_embeds_from_boundingbox<{{elem_type}}>(
        out,
        box,
        embed_dim,
        batch_size,
        num_boxes,
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
{{indent}}    {{box}},
{{indent}}    {{embed_dim}},
{{indent}}    {{batch_size}},
{{indent}}    {{num_boxes}},
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


def gen_function_call(func_attrs: Dict[str, Any], indent="  ") -> str:
    box = func_attrs["inputs"][0]
    out = func_attrs["outputs"][0]

    batch_size = box._attrs["shape"][0]
    num_boxes = box._attrs["shape"][1]

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        out=out._attrs["name"],
        box=box._attrs["name"],
        embed_dim=_to_int_expr(func_attrs["embed_dim"]),
        batch_size=_to_int_expr(batch_size),
        num_boxes=_to_int_expr(num_boxes),
        indent=indent,
    )


def gen_function(func_attrs, backend_spec):
    backend_spec = CUDASpec()
    elem_type = backend_spec.dtype_to_backend_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )
    return SRC_TEMPLATE.render(function_name=func_attrs["name"], elem_type=elem_type)


def gen_function_decl(func_attrs: Dict[str, Any], backend_spec) -> str:
    return FUNC_DECL_TEMPLATE.render(func_name=func_attrs["name"])


@registry.reg("cuda.get_fourier_embeds_from_boundingbox.gen_function")
def cuda_get_fourier_embeds_from_boundingbox_gen_function(func_attrs):
    return gen_function(func_attrs, CUDASpec())


@registry.reg("cuda.get_fourier_embeds_from_boundingbox.func_decl")
def cuda_get_fourier_embeds_from_boundingbox_gen_function_decl(
    func_attrs: Dict[str, Any],
) -> str:
    return gen_function_decl(func_attrs, CUDASpec())


@registry.reg("cuda.get_fourier_embeds_from_boundingbox.func_call")
def cuda_get_fourier_embeds_from_boundingbox_func_call(func_attrs, indent="  "):
    return gen_function_call(func_attrs, indent)
