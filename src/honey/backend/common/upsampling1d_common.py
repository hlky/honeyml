import jinja2

# pylint: disable=C0103,C0415,W0613,C0301,W0612


EXEC_TEMPLATE = jinja2.Template(
    """
{{indent}}upsampling1d_launcher(
{{indent}}    static_cast<const {{dtype}}*>(in_ptr),
{% if bias_add %}
  {{indent}}    static_cast<const {{dtype}}*>(res_ptr),
{% endif %}
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
{{header_files}}

namespace {
#define GPU_1D_KERNEL_LOOP(i, n) \
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

{% if mode == "bilinear" or mode == "linear" %}
__global__ void linear_upsampling1d_nhw_kernel(const {{dtype}}* input_raw,
                                                    {% if bias_add %}
                                                      const {{dtype}}* input_res_raw,
                                                    {% endif %}
                                                    {{dtype}}* output_raw,
                                                    const {{index_type}} batch,
                                                    const {{index_type}} in_width,
                                                    const {{index_type}} channels,
                                                    const {{index_type}} out_width) {
{% set vec_dtype = {"half": "half2", "float": "float2"}[dtype] %}
  const {{vec_dtype}}* input = (const {{vec_dtype}}*)input_raw;
{% if bias_add %}
  const {{vec_dtype}}* input_res = (const {{vec_dtype}}*)input_res_raw;
{% endif %}
  {{vec_dtype}}* output = ({{vec_dtype}}*)output_raw;

  {% if align_corners %}
  const float width_scale = in_width == 1 ? 0.0f : (float)(in_width - 1) / (out_width - 1);
  {% else %}
  const float width_scale = in_width / static_cast<float>(out_width);
  {% endif %}
  
  const int64_t num_threads = out_width * channels * batch;

GPU_1D_KERNEL_LOOP(out_idx, num_threads) {
    int64_t idx = out_idx;
    const int64_t c = idx % channels;
    idx /= channels;
    const int64_t x = idx % out_width;
    const int64_t b = idx / out_width;

    {% if align_corners %}
    const float in_x = x * width_scale;
    {% else %}
    const float in_x = (static_cast<float>(x) + 0.5f) * width_scale - 0.5f;
    {% endif %}
    const int64_t left_x_index = in_x > 0.0 ? floorf(in_x) : 0;
    const int64_t right_x_index =
        (in_x < in_width - 1) ? ceilf(in_x) : in_width - 1;
    const float x_lerp = in_x - floorf(in_x);

    const {{vec_dtype}} left = __ldg(
        input + (b * in_width + left_x_index) * channels + c);

    const {{vec_dtype}} right = __ldg(
        input + (b * in_width + right_x_index) * channels + c);

{% if dtype == "half" %}
    float x_val = __half2float(left{{half2_data_ref}}.x) + (__half2float(right{{half2_data_ref}}.x) - __half2float(left{{half2_data_ref}}.x)) * x_lerp;
    float y_val = __half2float(left{{half2_data_ref}}.y) + (__half2float(right{{half2_data_ref}}.y) - __half2float(left{{half2_data_ref}}.y)) * x_lerp;
{% elif dtype == "float" %}
    float x_val = left{{half2_data_ref}}.x + (right{{half2_data_ref}}.x - left{{half2_data_ref}}.x) * x_lerp;
    float y_val = left{{half2_data_ref}}.y + (right{{half2_data_ref}}.y - left{{half2_data_ref}}.y) * x_lerp;
{% endif %}

    float2 out = {0.f, 0.f};
    out.x = x_val;
    out.y = y_val;

{% if dtype == "half" %}
    {% if bias_add %}
      output[out_idx] = __hadd2(__float22half2_rn(out), __ldg(input_res + out_idx));
    {% else %}
      output[out_idx] = __float22half2_rn(out);
    {% endif %}
{% elif dtype == "float" %}
    {% if bias_add %}
      const auto tmp = __ldg(input_res + out_idx);
      out.x += tmp.x;
      out.y += tmp.y;
    {% endif %}
    output[out_idx] = out;
{% endif %}
  }

}

{% else %}
template <typename T, typename Telement, int element_in_Tio>
__global__ void nearest_upsampling1d_nhw_kernel(const T* input,
                                                    {% if bias_add %}
                                                      const T* input_res,
                                                    {% endif %}
                                                    T* output,
                                                    const {{index_type}} batch,
                                                    const {{index_type}} in_width,
                                                    const {{index_type}} channels,
                                                    const {{index_type}} out_width) {

  const float width_scale = in_width / static_cast<float>(out_width);
  const int64_t nthreads = out_width * channels * batch;

GPU_1D_KERNEL_LOOP(index, nthreads) {
    int n = index;
    int c = n % channels;
    n /= channels;
    int out_x = n % out_width;
    n /= out_width;

    const T* bottom_data_n = input + n * channels * in_width;
    const int in_x =
        max(min(static_cast<int>(
                    {% if mode == "nearest-exact"%}
                    floorf((static_cast<float>(out_x) + 0.5f) * width_scale)),
                    {% else %}
                    floorf((static_cast<float>(out_x)) * width_scale)),
                    {% endif %}
                static_cast<int>(in_width) - 1),
            0);
    const int idx = in_x * channels + c;


  {% if bias_add %}
    T input_val = __ldg(bottom_data_n + idx);
    T input_res_val = __ldg(input_res + index);
    {% if tsize == 1 %}
    output[index] = input_val + input_res_val;

    {% elif tsize == 8 and dtype == "half" %}
    T output_val;
    Telement* pack_y = reinterpret_cast<Telement*>(&output_val);
    Telement* pack_x = reinterpret_cast<Telement*>(&input_val);
    Telement* pack_res = reinterpret_cast<Telement*>(&input_res_val);
    for (int k = 0 ; k < element_in_Tio ; k++)
      pack_y[k] = pack_x[k] + pack_res[k];
    output[index] =  output_val;

    {% else %}
    T output_val;
    output_val{{half2_data_ref}}.x = input_val{{half2_data_ref}}.x + input_res_val{{half2_data_ref}}.x;
    output_val{{half2_data_ref}}.y = input_val{{half2_data_ref}}.y + input_res_val{{half2_data_ref}}.y;
    output[index] = output_val;
    {% endif %}
  {% else %}
    output[index] = __ldg(bottom_data_n + idx);
  {% endif %}

  }
}

{% endif %}

template <typename integer>
constexpr __host__ __device__ inline integer ceil_div(integer n, integer m) {
  return (n + m - 1) / m;
}

template<typename ELEM_T>
void upsampling1d_launcher(const ELEM_T* input,
                    {% if bias_add %}
                      const ELEM_T* input_res,
                    {% endif %}
                      ELEM_T* output,
                      const {{index_type}} N,
                      const {{index_type}} W,
                      const {{index_type}} C,
                      const {{index_type}} WO,
                      {{prefix}}Stream_t stream) {
    const int64_t output_size = N * (C) * WO;
    dim3 grid(std::min(
      ceil_div(static_cast<int64_t>(output_size), static_cast<int64_t>(512)),
      static_cast<int64_t>(4096)));
    dim3 block(512);

{% if mode == "bilinear" or mode == "linear" %}
    linear_upsampling1d_nhw_kernel<<<grid, block, 0, stream>>>(
      input,
      {% if bias_add %}
        input_res,
      {% endif %}
      output,
      N, W, C/2, WO);
{% else %}
  {% if dtype == "float" %}
    {% if tsize == 1 %}
    nearest_upsampling1d_nhw_kernel<float, float, 1><<<grid, block, 0, stream>>>(
      (const float*)input,
      {% if bias_add %}
        (const float*)input_res,
      {% endif %}
      (float*)output,
      N, W, C, WO);
    {% else %}
    nearest_upsampling1d_nhw_kernel<float2, float, 2><<<grid, block, 0, stream>>>(
      (const float2*)input,
      {% if bias_add %}
        (const float2*)input_res,
      {% endif %}
      (float2*)output,
      N, W, C / 2, WO);
    {% endif %}
  {% else %}
    {% if tsize == 1 %}
    nearest_upsampling1d_nhw_kernel<half, half, 1><<<grid, block, 0, stream>>>(
      (const half *)input,
      {% if bias_add %}
        (const half *)input_res,
      {% endif %}
      (half *)output,
      N, W, C, WO);
    {% elif tsize == 8 %}
    nearest_upsampling1d_nhw_kernel<float4, half, 8><<<grid, block, 0, stream>>>(
      (const float4 *)input,
      {% if bias_add %}
        (const float4 *)input_res,
      {% endif %}
      (float4 *)output,
      N, W, C/8, WO);
    {% else %}
    nearest_upsampling1d_nhw_kernel<half2, half, 2><<<grid, block, 0, stream>>>(
      (const half2 *)input,
      {% if bias_add %}
        (const half2 *)input_res,
      {% endif %}
      (half2 *)output,
      N, W, C/2, WO);
    {% endif %}
  {% endif %}
{% endif %}
}
} // namespace

void {{function_name}} (
    const void* in_ptr,
    {% if bias_add %}
    const void* res_ptr,
    {% endif %}
    void* out_ptr,
    {{index_type}}* batch,
    {{index_type}}* in_w,
    {{index_type}}* in_ch,
    {{index_type}}* out_batch,
    {{index_type}}* out_w,
    {{prefix}}Stream_t stream
) {

  {{shape_function}}

  {{exec_paths}}
  throw std::runtime_error(
      "Unsupported workload for this 1D upsampling specialization."
  );
}
"""
)


FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
  const void*,
  {% if bias_add %}
  const void*,
  {% endif %}
  void*,
  {{index_type}}*,
  {{index_type}}*,
  {{index_type}}*,
  {{index_type}}*,
  {{index_type}}*,
  {{prefix}}Stream_t
);
"""
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}    {{in_ptr}},
{% if bias_add %}
{{indent}}    {{res_ptr}},
{% endif %}
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


def gen_function_decl(func_attrs, backend_spec, bias_add=False):
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
        bias_add=bias_add,
    )


def gen_alignment(x):
    in_channel = x.shape()[-1].value()
    if in_channel % 8 == 0:
        tsize = 8
    elif in_channel % 4 == 0:
        tsize = 4
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
        r = func_attrs["inputs"][1]
        return FUNC_CALL_TEMPLATE.render(
            func_name=func_attrs["name"],
            index_type=backend_spec.index_type,
            in_ptr=x._attrs["name"],
            res_ptr=r._attrs["name"],
            out_ptr=y._attrs["name"],
            p_batch="&" + xshape[0]._attrs["name"],
            p_in_ch="&" + xshape[2]._attrs["name"],
            p_in_w="&" + xshape[1]._attrs["name"],
            p_out_batch="&" + yshape[0]._attrs["name"],
            p_out_w="&" + yshape[1]._attrs["name"],
            indent=indent,
            bias_add=bias_add,
        )
    else:
        return FUNC_CALL_TEMPLATE.render(
            func_name=func_attrs["name"],
            index_type=backend_spec.index_type,
            in_ptr=x._attrs["name"],
            out_ptr=y._attrs["name"],
            p_batch="&" + xshape[0]._attrs["name"],
            p_in_ch="&" + xshape[2]._attrs["name"],
            p_in_w="&" + xshape[1]._attrs["name"],
            p_out_batch="&" + yshape[0]._attrs["name"],
            p_out_w="&" + yshape[1]._attrs["name"],
            indent=indent,
            bias_add=bias_add,
        )
