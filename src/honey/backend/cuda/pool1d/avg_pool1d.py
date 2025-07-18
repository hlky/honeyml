import jinja2

from honey.backend import registry

from honey.backend.backend_spec import CUDASpec
from honey.backend.cuda.pool1d import pool1d

# pylint: disable=C0103,C0415,W0613,C0301,W0612


EXEC_TEMPLATE = jinja2.Template(
    """
{{indent}}avg_pool_launcher<{{dtype}}, {{kernel_size}}, {{stride}}, {{padding}}>(
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
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "cutlass/util/host_tensor.h"

namespace {

template <int kernel_size, int stride, int padding>
__global__ void avg_pool_nhwc_kernel(const {{dtype}}* input_raw,
                                     {{dtype}}* output_raw,
                                     const int N,
                                     const int W,
                                     const int C,
                                     const int WO) {
{% set vec_dtype = {"half": "half2", "float": "float2"}[dtype] %}
  const {{vec_dtype}}* input = (const {{vec_dtype}}*)input_raw;
  {{vec_dtype}}* output = ({{vec_dtype}}*)output_raw;

  const int tid = threadIdx.x;
  const int n_idx = blockIdx.x;
  const int out_w_idx = blockIdx.y;

  int w_start_idx = out_w_idx * stride - padding;
  int w_end_idx = w_start_idx + kernel_size;
  w_start_idx = (w_start_idx < 0) ? 0 : w_start_idx;
  w_end_idx = w_end_idx > W ? W : w_end_idx;

  input += n_idx * W * C;
  output += (n_idx * WO + out_w_idx) * C;
  const float norm_factor =
      static_cast<float>(1.0f / kernel_size);
  for (int c_idx = tid; c_idx < C; c_idx += blockDim.x) {
    float2 avg = {0.f, 0.f};
    for (int w = w_start_idx; w < w_end_idx; w++) {
      const int idx = w * C;
      const {{vec_dtype}} tmp = __ldg(input + (idx + c_idx));
{% if dtype == "half" %}
      avg.x += __half2float(tmp.x);
      avg.y += __half2float(tmp.y);
{% else %}
      avg.x += tmp.x;
      avg.y += tmp.y;
{% endif %}
    }

    avg.x *= norm_factor;
    avg.y *= norm_factor;
{% if dtype == "half" %}
    output[c_idx] = __float22half2_rn(avg);
{% else %}
    output[c_idx] = avg;
{% endif %}
  }
}

template <typename ElemT, int kernel_size, int stride, int padding>
void avg_pool_launcher(const ElemT* input,
                       ElemT* output,
                       const int N,
                       const int W,
                       const int C,
                       const int WO,
                       cudaStream_t stream)
{
  int num_thread = C / 2;
  if (num_thread > 256) {
      num_thread = 256;
  } else if (num_thread == 0) {
      num_thread = 1;
  }
  dim3 grid(N, WO);
  dim3 block(num_thread);
  avg_pool_nhwc_kernel<kernel_size, stride, padding>
      <<<grid, block, 0, stream>>>(input, output, N, W,
                                   C / 2, WO);
}
} // namespace

void {{function_name}} (
    const void* in_ptr,
    void* out_ptr,
    int64_t* batch,
    int64_t* in_w,
    int64_t* in_ch,
    int64_t* out_batch,
    int64_t* out_w,
    cudaStream_t stream
) {
  {{shape_function}}
  {{exec_paths}}
  throw std::runtime_error(
      "Unsupported workload for this avg pool1d specialization."
  );
}
"""
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
        kernel_h=func_attrs["kernel_size"],
        kernel_w=func_attrs["kernel_size"],  # This might be irrelevant for 1D
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
    for key in exec_path:
        program = EXEC_TEMPLATE.render(
            indent=" " * 4,
            kernel_size=func_attrs["kernel_size"],
            padding=func_attrs["pad"],
            stride=func_attrs["stride"],
            dtype=dtype,
        )
        exec_inst = exec_cond_template.render(indent="  ", cond=key, program=program)
        exec_paths += exec_inst
    return SRC_TEMPLATE.render(
        function_name=func_name,
        shape_function=shape_func,
        exec_paths=exec_paths,
        dtype=dtype,
    )


@registry.reg("cuda.avg_pool1d.func_decl")
def avg_pool1d_gen_function_decl(func_attrs):
    func_name = func_attrs["name"]
    return pool1d.gen_function_decl(func_name)


@registry.reg("cuda.avg_pool1d.func_call")
def avg_pool1d_gen_function_call(func_attrs, indent="  "):
    return pool1d.gen_function_call(func_attrs, indent)
