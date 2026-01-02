from typing import Any, Dict
from dinoml.backend import registry
from dinoml.backend.backend_spec import CUDASpec
import jinja2

SRC = jinja2.Template(
    """
#include <dinoml/device.h>
#include <ops/fir_upsample2d.h>

void {{name}}(
    void* out,
    const void* in,
    int N, int H, int W, int C, int up, int pad0, int pad1,
    dinoml::DeviceStream stream
) {
    invoke_fir_upsample2d<{{dtype}}>(out, in, N, H, W, C, up, pad0, pad1, stream);
}
"""
)

DECL = jinja2.Template(
    """
void {{name}}(void*, const void*, int, int, int, int, int, int, int, dinoml::DeviceStream);
"""
)


FUNC_CALL_TEMPLATE = jinja2.Template(
    r"""
{{indent}}{{func_name}}(
{{indent}}    {{output}},
{{indent}}    {{input}},
{{indent}}    {{N}},
{{indent}}    {{H}},
{{indent}}    {{W}},
{{indent}}    {{C}},
{{indent}}    {{up}},
{{indent}}    {{pad0}},
{{indent}}    {{pad1}},
{{indent}}    stream
{{indent}});
"""
)


@registry.reg("cuda.fir_upsample2d.gen_function")
def gen_func(attrs):
    dtype = CUDASpec().dtype_to_backend_type(attrs["outputs"][0]._attrs["dtype"])
    return SRC.render(name=attrs["name"], dtype=dtype)


@registry.reg("cuda.fir_upsample2d.func_decl")
def gen_decl(attrs):
    return DECL.render(name=attrs["name"])


def gen_call(func_attrs: Dict[str, Any], indent="  ") -> str:
    x = func_attrs["inputs"][0]
    y = func_attrs["outputs"][0]
    # NHWC: shape [N,H,W,C]
    N, H, W, C = x._attrs["shape"]
    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=y._attrs["name"],
        input=x._attrs["name"],
        N=N._attrs["name"],
        H=H._attrs["name"],
        W=W._attrs["name"],
        C=C._attrs["name"],
        up=func_attrs["up"],
        pad0=func_attrs["pad0"],
        pad1=func_attrs["pad1"],
        indent=indent,
    )


@registry.reg("cuda.fir_upsample2d.func_call")
def cuda_fir_downsample2d_func_call(func_attrs, indent="  "):
    return gen_call(func_attrs, indent)
