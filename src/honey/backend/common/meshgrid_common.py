import jinja2

SRC_TEMPLATE = jinja2.Template(
    """
{{header_files}}

namespace {

template <typename T>
__global__ void meshgrid_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    {{index_type}} dim,
    {{index_type}} out_stride,
    {{index_type}} num_elements
) {
    {{index_type}} idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        {{index_type}} d = (idx / out_stride) % dim;
        output[idx] = input[d];
    }
}

}  // namespace

void {{function_name}}(
    std::array<const void*, {{num_inputs}}> in_ptrs,
    std::array<void*, {{num_inputs}}> out_ptrs,
    std::array<{{index_type}}, {{num_inputs}}> lens,
    {{index_type}} num_inputs,
    {{prefix}}Stream_t stream
) {
    {{index_type}} num_elements = 1;
    for ({{index_type}} i = 0; i < num_inputs; ++i) {
        num_elements *= lens[i];
    }

    for ({{index_type}} i = 0; i < num_inputs; ++i) {
        {{index_type}} out_stride = 1;
        for ({{index_type}} j = i + 1; j < num_inputs; ++j) {
            out_stride *= lens[j];
        }
        {{index_type}} threads_per_block = 256;
        {{index_type}} num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

        meshgrid_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
            static_cast<const {{elem_input_type}}*>(in_ptrs[i]),
            static_cast<{{elem_output_type}}*>(out_ptrs[i]),
            lens[i],
            out_stride,
            num_elements
        );
    }
}
"""
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
    std::array<const void*, {{num_inputs}}>,
    std::array<void*, {{num_inputs}}>,
    std::array<{{index_type}}, {{num_inputs}}>,
    {{index_type}},
    {{prefix}}Stream_t
);
"""
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{in_ptrs}},
{{indent}}    {{out_ptrs}},
{{indent}}    {{lens}},
{{indent}}    {{num_inputs}},
{{indent}}    stream
{{indent}});
"""
)


def gen_function_decl(func_attrs, backend_spec):
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
    x = func_attrs["inputs"][0]
    num_inputs = len(func_attrs["inputs"])
    return FUNC_DECL_TEMPLATE.render(
        index_type=backend_spec.index_type,
        prefix=backend_spec.prefix,
        func_name=func_attrs["name"],
        dtype=backend_spec.dtype_to_backend_type(x._attrs["dtype"]),
        num_inputs=num_inputs,
    )


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
    input_tensors = func_attrs["inputs"]
    output_tensors = func_attrs["outputs"]
    num_inputs = len(input_tensors)
    indexing = func_attrs.get("indexing", "xy")
    if indexing == "xy":
        input_tensors.reverse()
        output_tensors.reverse()
    in_ptrs = "{" + ", ".join([tensor._attrs["name"] for tensor in input_tensors]) + "}"
    out_ptrs = (
        "{" + ", ".join([tensor._attrs["name"] for tensor in output_tensors]) + "}"
    )
    lens = (
        "{"
        + ", ".join(
            [str(tensor._attrs["shape"][0]._attrs["name"]) for tensor in input_tensors]
        )
        + "}"
    )
    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        in_ptrs=in_ptrs,
        out_ptrs=out_ptrs,
        lens=lens,
        num_inputs=num_inputs,
        indent=indent,
    )
