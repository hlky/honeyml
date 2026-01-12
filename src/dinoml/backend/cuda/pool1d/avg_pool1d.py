import jinja2

from dinoml.backend import registry

from dinoml.backend.backend_spec import CUDASpec


EXEC_TEMPLATE = jinja2.Template(
    """
{{indent}}avg_pool_1d_launcher<{{dtype}}, {{vec_type}}, {{kernel_size}}, {{stride}}, {{padding}}>(
{{indent}}    static_cast<const {{dtype}}*>(in_ptr),
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
#include <ops/avg_pool_1d.h>

void {{function_name}} (
    const void* in_ptr,
    void* out_ptr,
    int64_t* batch,
    int64_t* in_w,
    int64_t* in_ch,
    int64_t* out_batch,
    int64_t* out_w,
    dinoml::DeviceStream stream
) {
  {{shape_function}}
  {{exec_paths}}
  throw std::runtime_error(
      "Unsupported workload for this avg pool1d specialization."
  );
}
"""
)


FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
  const void*,
  void*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  int64_t*,
  dinoml::DeviceStream
);
"""
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{in_ptr}},
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


def gen_function_decl(func_name):
    return FUNC_DECL_TEMPLATE.render(func_name=func_name)


def gen_function_call(func_attrs, indent="  "):
    x = func_attrs["inputs"][0]
    xshape = x._attrs["shape"]
    y = func_attrs["outputs"][0]
    yshape = y._attrs["shape"]
    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        in_ptr=x._attrs["name"],
        out_ptr=y._attrs["name"],
        p_batch="&" + xshape[0]._attrs["name"],
        p_in_ch="&" + xshape[2]._attrs["name"],
        p_in_w="&" + xshape[1]._attrs["name"],
        p_out_batch="&" + yshape[0]._attrs["name"],
        p_out_w="&" + yshape[1]._attrs["name"],
        indent=indent,
    )


@registry.reg("cuda.avg_pool1d.gen_function")
def gen_function(
    func_attrs,
    exec_cond_template,
    shape_eval_template,
    shape_save_template,
):
    func_name = func_attrs["name"]
    exec_path = func_attrs["exec_path"]
    backend_spec = CUDASpec()
    dtype = backend_spec.dtype_to_backend_type(func_attrs["inputs"][0]._attrs["dtype"])
    shape_eval_func = shape_eval_template.render(
        indent="  ",
        dtype="int64_t ",
        x_dim0="*batch",
        x_dim1="*in_w",
        x_dim2="*in_ch",
        kernel_w=func_attrs["kernel_size"],
        stride=func_attrs["stride"],
        pad=func_attrs["pad"],
        div="/",
    )
    shape_save_func = shape_save_template.render(
        indent="  ",
        y_dim0="*out_batch",
        y_dim1="*out_w",
    )
    shape_func = shape_eval_func + shape_save_func
    exec_paths = ""
    dtype_vec_type = {"float": "float2", "half": "half2", "bfloat16": "bfloat162"}
    vec_type = dtype_vec_type[dtype]
    for key in exec_path:
        program = EXEC_TEMPLATE.render(
            indent=" " * 4,
            kernel_size=func_attrs["kernel_size"],
            padding=func_attrs["pad"],
            stride=func_attrs["stride"],
            dtype=dtype,
            vec_type=vec_type,
        )
        exec_inst = exec_cond_template.render(indent="  ", cond=key, program=program)
        exec_paths += exec_inst
    return SRC_TEMPLATE.render(
        function_name=func_name,
        shape_function=shape_func,
        exec_paths=exec_paths,
    )


@registry.reg("cuda.avg_pool1d.func_decl")
def avg_pool1d_gen_function_decl(func_attrs):
    func_name = func_attrs["name"]
    return gen_function_decl(func_name)


@registry.reg("cuda.avg_pool1d.func_call")
def avg_pool1d_gen_function_call(func_attrs, indent="  "):
    return gen_function_call(func_attrs, indent)
