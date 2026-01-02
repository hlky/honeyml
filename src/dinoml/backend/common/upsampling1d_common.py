import jinja2

# pylint: disable=C0103,C0415,W0613,C0301,W0612


EXEC_TEMPLATE = jinja2.Template(
    """
{{indent}}upsampling_1d_launcher<{{dtype}}, {{vec_type}}, int64_t, {{alignment}}, {{mode}}, {{align_corners}}, {{exact}}, {{has_residual}}>(
{{indent}}    static_cast<const {{dtype}}*>(in_ptr),
{{indent}}    static_cast<const {{dtype}}*>(res_ptr),
{{indent}}    static_cast<{{dtype}}*>(out_ptr),
{{indent}}    NI,
{{indent}}    WI,
{{indent}}    CI,
{{indent}}    WO,
{{indent}}    stream
{{indent}});
{{indent}}return;
"""
)

SRC_TEMPLATE = jinja2.Template(
    """
#include <dinoml/device.h>
#include <ops/upsampling_1d.h>

void {{function_name}} (
    const void* in_ptr,
    const void* res_ptr,
    void* out_ptr,
    {{index_type}}* batch,
    {{index_type}}* in_w,
    {{index_type}}* in_ch,
    {{index_type}}* out_batch,
    {{index_type}}* out_w,
    dinoml::DeviceStream stream
) {

  {{shape_function}}

  {{exec_paths}}
  throw std::runtime_error(
      "Unsupported workload for this upsampling 1d specialization."
  );
}
"""
)


FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
  const void*,
  const void*,
  void*,
  {{index_type}}*,
  {{index_type}}*,
  {{index_type}}*,
  {{index_type}}*,
  {{index_type}}*,
  dinoml::DeviceStream
);
"""
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{in_ptr}},
{{indent}}    {{res_ptr}},
{{indent}}    {{out_ptr}},
{{indent}}    {{p_batch}},
{{indent}}    {{p_in_w}},
{{indent}}    {{p_in_ch}},
{{indent}}    {{p_out_batch}},
{{indent}}    {{p_out_w}},
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
        index_type=backend_spec.index_type,
        prefix=backend_spec.prefix,
        func_name=func_attrs["name"],
    )


def gen_alignment(x):
    in_channel = x.shape()[-1].value()
    if in_channel % 8 == 0:
        tsize = 8
    elif in_channel % 4 == 0:
        tsize = 2
    elif in_channel % 2 == 0:
        tsize = 2
    else:
        tsize = 1
    return tsize


def gen_function_call(func_attrs, backend_spec, indent="  ", bias_add=False):
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
    xshape = x._attrs["shape"]
    y = func_attrs["outputs"][0]
    yshape = y._attrs["shape"]
    if bias_add:
        r = func_attrs["inputs"][1]._attrs["name"]
    else:
        r = "nullptr"
    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        index_type=backend_spec.index_type,
        in_ptr=x._attrs["name"],
        res_ptr=r,
        out_ptr=y._attrs["name"],
        p_batch="&" + xshape[0]._attrs["name"],
        p_in_ch="&" + xshape[2]._attrs["name"],
        p_in_w="&" + xshape[1]._attrs["name"],
        p_out_batch="&" + yshape[0]._attrs["name"],
        p_out_w="&" + yshape[1]._attrs["name"],
        indent=indent,
        bias_add=bias_add,
    )


def gen_function(
    func_attrs,
    template_path,
    exec_cond_template,
    shape_eval_template,
    shape_save_template,
    backend_spec,
    bias_add=False,
):
    func_name = func_attrs["name"]
    exec_path = func_attrs["exec_path"]
    x = func_attrs["inputs"][0]
    input_type = backend_spec.dtype_to_backend_type(x._attrs["dtype"])
    alignment = gen_alignment(x)
    if func_attrs["mode"] == "linear":
        if alignment > 1:
            alignment = 2
        if input_type == "float":
            vec_type = "float2"
        elif input_type == "bfloat16":
            vec_type = "bfloat162"
        else:
            vec_type = "half2"
    else:
        if input_type == "float":
            if alignment > 1:
                vec_type = "float2"
                alignment = 2
            else:
                vec_type = "float"
        elif input_type == "half":
            if alignment == 8:
                vec_type = "float4"
            elif 1 < alignment < 8:
                vec_type = "half2"
            else:
                vec_type = "half"
        elif input_type == "bfloat16":
            if alignment == 8:
                vec_type = "float4"
            elif 1 < alignment < 8:
                vec_type = "bfloat162"
            else:
                vec_type = "bfloat16"

    args = {
        "indent": "    ",
        "dtype": "int64_t ",
        "div": "/",
        "x_dim0": "*batch",
        "x_dim1": "*in_w",
        "x_dim2": "*in_ch",
    }
    if func_attrs["out_shape"] is True:
        args["out_w"] = "*out_w"
    else:
        args["scale_factor"] = func_attrs["scale_factor"]
    shape_eval_func = shape_eval_template.render(
        **args,
    )
    shape_save_func = shape_save_template.render(
        indent="  ",
        y_dim0="*out_batch",
        y_dim1="*out_w",
    )
    shape_func = shape_eval_func + shape_save_func
    exec_paths = ""

    upsampling_mode = {
        "linear": "dinoml::Upsampling1dMode::LINEAR",
        "nearest": "dinoml::Upsampling1dMode::NEAREST",
        "nearest-exact": "dinoml::Upsampling1dMode::NEAREST_EXACT",
    }
    mode = upsampling_mode[func_attrs["mode"]]
    if func_attrs["align_corners"] is None:
        align_corners = "false"
    else:
        align_corners = str(func_attrs["align_corners"]).lower()
    exact = str(func_attrs["mode"] == "nearest-exact").lower()
    has_residual = str(bias_add).lower()
    for key in exec_path:
        program = EXEC_TEMPLATE.render(
            dtype=input_type,
            vec_type=vec_type,
            alignment=alignment,
            mode=mode,
            align_corners=align_corners,
            exact=exact,
            has_residual=has_residual,
        )
        exec_inst = exec_cond_template.render(indent="  ", cond=key, program=program)
        exec_paths += exec_inst
    return SRC_TEMPLATE.render(
        function_name=func_name,
        shape_function=shape_func,
        exec_paths=exec_paths,
        index_type=backend_spec.index_type,
    )
