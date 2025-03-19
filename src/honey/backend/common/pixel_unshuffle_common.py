import jinja2

SRC_TEMPLATE = jinja2.Template(
    """
{{header_files}}

namespace {

__global__ void pixel_unshuffle_kernel(
    const {{elem_input_type}}* input,
    {{elem_output_type}}* output,
    {{index_type}} batch,
    {{index_type}} in_h,
    {{index_type}} in_w,
    {{index_type}} in_ch,
    {{index_type}} r
) {
    {{index_type}} idx = blockIdx.x * blockDim.x + threadIdx.x;
    {{index_type}} total_elements = batch * in_h * in_w * in_ch;
    if (idx < total_elements) {
        {{index_type}} out_ch = in_ch * r * r;
        {{index_type}} n = idx / (in_h * in_w * in_ch);
        {{index_type}} c = (idx / (in_h * in_w)) % in_ch;
        {{index_type}} h = (idx / in_w) % in_h;
        {{index_type}} w = idx % in_w;

        {{index_type}} rh = h % r;
        {{index_type}} rw = w % r;
        {{index_type}} out_h = h / r;
        {{index_type}} out_w = w / r;

        {{index_type}} in_index = n * in_h * in_w * in_ch + h * in_w * in_ch + w * in_ch + c;
        {{index_type}} out_index = n * (in_h / r) * (in_w / r) * out_ch + out_h * (in_w / r) * out_ch + out_w * out_ch + (c * r * r) + (rh * r) + rw;

        output[out_index] = input[in_index];
    }
}

}  // namespace

void {{function_name}}(
    const void* in_ptr,
    void* out_ptr,
    {{index_type}}* batch,
    {{index_type}}* in_h,
    {{index_type}}* in_w,
    {{index_type}}* in_ch,
    {{index_type}}* out_batch,
    {{index_type}}* out_h,
    {{index_type}}* out_w,
    {{index_type}}* out_ch,
    {{index_type}} r,
    {{prefix}}Stream_t stream
) {
    {{index_type}} total_elements = (*batch) * (*in_h) * (*in_w) * (*in_ch);
    {{index_type}} threads_per_block = 256;
    {{index_type}} num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    pixel_unshuffle_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        static_cast<const {{elem_input_type}}*>(in_ptr),
        static_cast<{{elem_output_type}}*>(out_ptr),
        *batch,
        *in_h,
        *in_w,
        *in_ch,
        r
    );
}
"""
)

EXEC_TEMPLATE = jinja2.Template(
    """
{{indent}}pixel_unshuffle_launcher(
{{indent}}    static_cast<const {{dtype}}*>(in_ptr),
{{indent}}    static_cast<{{dtype}}*>(out_ptr),
{{indent}}    NI,
{{indent}}    HI,
{{indent}}    WI,
{{indent}}    CI,
{{indent}}    r,
{{indent}}    stream
{{indent}});
{{indent}}return;
"""
)

FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
    const void*,
    void*,
    {{index_type}}*,
    {{index_type}}*,
    {{index_type}}*,
    {{index_type}}*,
    {{index_type}}*,
    {{index_type}}*,
    {{index_type}}*,
    {{index_type}}*,
    {{index_type}},
    {{prefix}}Stream_t
);
"""
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{in_ptr}},
{{indent}}    {{out_ptr}},
{{indent}}    {{p_batch}},
{{indent}}    {{p_in_h}},
{{indent}}    {{p_in_w}},
{{indent}}    {{p_in_ch}},
{{indent}}    {{p_out_batch}},
{{indent}}    {{p_out_h}},
{{indent}}    {{p_out_w}},
{{indent}}    {{p_out_ch}},
{{indent}}    {{r}},
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
    return FUNC_DECL_TEMPLATE.render(
        func_name=func_attrs["name"],
        prefix=backend_spec.prefix,
        index_type=backend_spec.index_type,
        dtype=backend_spec.dtype_to_backend_type(
            func_attrs["inputs"][0]._attrs["dtype"]
        ),
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
    x = func_attrs["inputs"][0]
    y = func_attrs["outputs"][0]
    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        in_ptr=x._attrs["name"],
        out_ptr=y._attrs["name"],
        p_batch="&" + x._attrs["shape"][0]._attrs["name"],
        p_in_h="&" + x._attrs["shape"][1]._attrs["name"],
        p_in_w="&" + x._attrs["shape"][2]._attrs["name"],
        p_in_ch="&" + x._attrs["shape"][3]._attrs["name"],
        p_out_batch="&" + y._attrs["shape"][0]._attrs["name"],
        p_out_h="&" + y._attrs["shape"][1]._attrs["name"],
        p_out_w="&" + y._attrs["shape"][2]._attrs["name"],
        p_out_ch="&" + y._attrs["shape"][3]._attrs["name"],
        r=func_attrs["r"],
        indent=indent,
    )
