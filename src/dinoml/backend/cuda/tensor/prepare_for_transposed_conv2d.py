import jinja2
from dinoml.backend import registry
from dinoml.backend.backend_spec import CUDASpec


SRC_TEMPLATE = jinja2.Template(
    r"""
#include <dinoml/device.h>
#include <ops/prepare_for_transposed_conv2d.h>

void {{function_name}}(
    void* out,
    const void* in,
    int64_t n,
    int64_t h,
    int64_t w,
    int64_t c,
    int64_t stride_h,
    int64_t stride_w,
    dinoml::DeviceStream stream
) {
    dinoml::invoke_prepare_for_transposed_conv2d<{{dtype}}>(
        out,
        in,
        n,
        h,
        w,
        c,
        stride_h,
        stride_w,
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
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    int64_t,
    dinoml::DeviceStream
);
"""
)


FUNC_CALL_TEMPLATE = jinja2.Template(
    r"""
{{indent}}{{func_name}}(
{{indent}}    {{out}},
{{indent}}    {{inp}},
{{indent}}    {{n}},
{{indent}}    {{h}},
{{indent}}    {{w}},
{{indent}}    {{c}},
{{indent}}    {{stride_h}},
{{indent}}    {{stride_w}},
{{indent}}    stream
{{indent}});
"""
)


@registry.reg("cuda.prepare_for_transposed_conv2d.gen_function")
def gen_function(func_attrs):
    backend = CUDASpec()
    dtype = backend.dtype_to_backend_type(func_attrs["outputs"][0]._attrs["dtype"])
    return SRC_TEMPLATE.render(
        function_name=func_attrs["name"],
        dtype=dtype,
    )


@registry.reg("cuda.prepare_for_transposed_conv2d.func_decl")
def gen_decl(func_attrs):
    return FUNC_DECL_TEMPLATE.render(func_name=func_attrs["name"])


@registry.reg("cuda.prepare_for_transposed_conv2d.func_call")
def gen_call(func_attrs, indent="  "):
    x = func_attrs["inputs"][0]
    stride_h, stride_w = func_attrs["stride"]
    shape = x._attrs["shape"]

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        out=func_attrs["outputs"][0]._attrs["name"],
        inp=x._attrs["name"],
        n=shape[0]._attrs["name"],
        h=shape[1]._attrs["name"],
        w=shape[2]._attrs["name"],
        c=shape[3]._attrs["name"],
        stride_h=stride_h,
        stride_w=stride_w,
        indent=indent,
    )
