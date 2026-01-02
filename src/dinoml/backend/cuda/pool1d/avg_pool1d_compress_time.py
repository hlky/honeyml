import jinja2
from dinoml.backend import registry
from dinoml.backend.backend_spec import CUDASpec

SRC = jinja2.Template(
    r"""
#include <dinoml/device.h>
#include <ops/avg_pool1d_compress_time.h>
#include <stdexcept>

void {{function_name}}(
    const void* in_ptr,
    void* out_ptr,
    int64_t* batch,
    int64_t* in_w,
    int64_t* in_ch,
    int64_t* out_batch,
    int64_t* out_w,
    dinoml::DeviceStream stream
) {
  const int64_t N = *batch;     // N = batch_size * H * W
  const int64_t W = *in_w;      // frames
  const int64_t C = *in_ch;     // channels

  // Output W matches reference:
  // if W odd and >1: 1 + (W-1)/2 else W/2, with W==1 -> 1
  int64_t WO;
  if (W <= 1) WO = 1;
  else if ((W & 1) == 1) WO = 1 + (W - 1) / 2;
  else WO = W / 2;

  *out_batch = N;
  *out_w = WO;

  dinoml::avg_pool1d_compress_time_launch<{{dtype}}>(
      in_ptr,
      out_ptr,
      N,
      C,
      W,
      stream
  );
}
"""
)

DECL = jinja2.Template(
    r"""
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

CALL = jinja2.Template(
    r"""
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


@registry.reg("cuda.avg_pool1d_compress_time.gen_function")
def gen_function(
    func_attrs,
    exec_cond_template,
    shape_eval_template,
    shape_save_template,
):
    spec = CUDASpec()
    dtype = spec.dtype_to_backend_type(func_attrs["inputs"][0]._attrs["dtype"])
    return SRC.render(function_name=func_attrs["name"], dtype=dtype)


@registry.reg("cuda.avg_pool1d_compress_time.func_decl")
def gen_decl(func_attrs):
    return DECL.render(func_name=func_attrs["name"])


@registry.reg("cuda.avg_pool1d_compress_time.func_call")
def gen_call(func_attrs, indent="  "):
    x = func_attrs["inputs"][0]
    y = func_attrs["outputs"][0]
    xshape = x._attrs["shape"]
    yshape = y._attrs["shape"]

    return CALL.render(
        func_name=func_attrs["name"],
        in_ptr=x._attrs["name"],
        out_ptr=y._attrs["name"],
        p_batch="&" + xshape[0]._attrs["name"],
        p_in_ch="&" + xshape[1]._attrs["name"],
        p_in_w="&" + xshape[2]._attrs["name"],
        p_out_batch="&" + yshape[0]._attrs["name"],
        p_out_w="&" + yshape[2]._attrs["name"],
        indent=indent,
    )
