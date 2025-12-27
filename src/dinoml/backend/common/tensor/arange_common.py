from typing import Any, Dict

import jinja2

from dinoml.compiler.base import IntVar

KERNEL_TEMPLATE = jinja2.Template(
    """
template <typename T>
__global__ void ArangeKernel(T* output, T start, T step, int64_t numel) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = start + step * (T)idx;
    }
}

void invoke_arange(
    {{elem_type}}* output,
    {{elem_type}} start,
    {{elem_type}} step,
    int64_t numel,
    {{prefix}}Stream_t stream) {
    if (numel < 1024) {
        dim3 grid(1);
        dim3 block(numel);
        ArangeKernel<{{elem_type}}><<<grid, block, 0, stream>>>(output, start, step, numel);
    } else {
        dim3 grid((numel + 1024 - 1) / 1024);
        dim3 block(1024);
        ArangeKernel<{{elem_type}}><<<grid, block, 0, stream>>>(output, start, step, numel);
    }
}
    """
)

FUNC_TEMPLATE = jinja2.Template(
    """
{{header_files}}

namespace {

{{kernel}}

}  // namespace

{{func_signature}}
{
    invoke_arange(
        static_cast<{{elem_type}}*>(output),
        start,
        step,
        num_elements,
        stream);
}
    """
)

FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(void* output,
                   {{elem_type}} start,
                   {{elem_type}} step,
                   int64_t num_elements,
                   {{prefix}}Stream_t stream)
    """
)

FUNC_DECL = jinja2.Template(
    """
    {{func_signature}};
    """
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}   {{output}}, {{start}}, {{step}}, {{num_elements}}, stream /* default stream */
{{indent}});
    """
)


def gen_function_call(func_attrs: Dict[str, Any], indent="  ") -> str:
    output_name = func_attrs["outputs"][0]._attrs["name"]
    start: IntVar = func_attrs["start"]
    step: IntVar = func_attrs["step"]
    num_elements = func_attrs["outputs"][0].shape()[0]

    start_value = start._attrs["name"]
    if start_value is None:
        start_value = start.symbolic_value()
    step_value = step._attrs["name"]
    if step_value is None:
        step_value = step.symbolic_value()
    num_elements_value = num_elements._attrs["name"]

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=output_name,
        start=start_value,
        step=step_value,
        num_elements=num_elements_value,
        indent=indent,
    )


def gen_function(func_attrs: Dict[str, Any], header_files: str, backend_spec) -> str:
    output_y = func_attrs["outputs"][0]
    output_type = backend_spec.dtype_to_backend_type(output_y._attrs["dtype"])

    if output_y._attrs["dtype"] not in [
        "float32",
        "float16",
        "bfloat16",
        "int",
        "int64",
    ]:
        raise NotImplementedError("Unsupported data type for arange " + output_type)

    prefix = backend_spec.prefix

    return FUNC_TEMPLATE.render(
        header_files=header_files,
        elem_type=output_type,
        kernel=KERNEL_TEMPLATE.render(
            prefix=prefix,
            elem_type=output_type,
        ),
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"], prefix=prefix, elem_type=output_type
        ),
    )


def gen_function_decl(func_attrs: Dict[str, Any], backend_spec) -> str:
    output_type = backend_spec.dtype_to_backend_type(
        func_attrs["outputs"][0]._attrs["dtype"]
    )

    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"],
            prefix=backend_spec.prefix,
            elem_type=output_type,
        ).strip()
    )
