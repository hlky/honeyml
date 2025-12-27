from typing import Any, Dict

import jinja2
from dinoml.backend.common.elementwise_common import get_stride_expressions
from dinoml.frontend import IntVar

KERNEL_TEMPLATE = jinja2.Template(
    """
template <typename T, int64_t Rank>
__global__ void RepeatInterleaveKernel(
    T* output,
    const T* input,
    const int64_t repeats,
    const int64_t repeat_dim,
    std::array<int64_t, Rank> out_shape,
    std::array<int64_t, Rank> in_strides,
    const int64_t numel) {
  int64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < numel) {
    int64_t temp_tid = tid;
    int64_t input_idx = 0;
    int64_t factor = 1;

    for (int64_t i = Rank - 1; i >= 0; --i) {
      int64_t current_dim_idx = (temp_tid / factor) % out_shape[i];
      if (i == repeat_dim) {
        current_dim_idx /= repeats;
      }
      input_idx += current_dim_idx * in_strides[i];
      factor *= out_shape[i];
    }

    output[tid] = input[input_idx];
  }
}

void invoke_repeat_interleave(
    {{elem_type}}* output,
    const {{elem_type}}* input,
    const int64_t repeats,
    const int64_t repeat_dim,
    const int64_t numel,
    std::array<int64_t, {{rank}}> out_shape,
    std::array<int64_t, {{rank}}> in_strides,
    {{prefix}}Stream_t stream) {
  if (numel < 1024) {
    dim3 grid(1);
    dim3 block(numel);
    RepeatInterleaveKernel<{{elem_type}}, {{rank}}><<<grid, block, 0, stream>>>(
        output, input, repeats, repeat_dim, out_shape, in_strides, numel);
  } else {
    dim3 grid((numel + 1024 - 1) / 1024);
    dim3 block(1024);
    RepeatInterleaveKernel<{{elem_type}}, {{rank}}><<<grid, block, 0, stream>>>(
        output, input, repeats, repeat_dim, out_shape, in_strides, numel);
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
    invoke_repeat_interleave(
        static_cast<{{elem_type}}*>(output),
        static_cast<const {{elem_type}}*>(input),
        repeats,
        repeat_dim,
        num_elements,
        out_shape,
        in_strides,
        stream);
}
    """
)

FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(void* output,
                   const void* input,
                   const int64_t repeats,
                   const int64_t repeat_dim,
                   const int64_t num_elements,
                   std::array<int64_t, {{rank}}> out_shape,
                   std::array<int64_t, {{rank}}> in_strides,
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
{{indent}}   {{output}}, {{input}}, {{repeats}}, {{repeat_dim}}, {{num_elements}}, { {% for dim in out_shape %}{{dim}}, {% endfor %} }, { {% for stride in in_strides %}{{stride}}, {% endfor %} }, stream /* default stream */
{{indent}});
    """
)


def gen_function_call(func_attrs: Dict[str, Any], indent="  ") -> str:
    assert len(func_attrs["outputs"]) == 1
    assert len(func_attrs["inputs"]) == 1

    output_name = func_attrs["outputs"][0]._attrs["name"]
    input_name = func_attrs["inputs"][0]._attrs["name"]
    repeats = func_attrs["repeats"]
    if isinstance(repeats, IntVar):
        repeats = repeats._attrs["name"]
    repeat_dim = func_attrs["repeat_dim"]

    out_shape = [dim._attrs["name"] for dim in func_attrs["outputs"][0].shape()]
    in_strides = get_stride_expressions(func_attrs["inputs"][0].shape()) + ["1"]
    num_elements = " * ".join(out_shape)
    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=output_name,
        input=input_name,
        repeats=repeats,
        repeat_dim=repeat_dim,
        out_shape=out_shape,
        in_strides=in_strides,
        num_elements=num_elements,
        indent=indent,
    )


def gen_function(func_attrs: Dict[str, Any], header_files: str, backend_spec) -> str:
    input_x = func_attrs["inputs"][0]
    output_y = func_attrs["outputs"][0]
    input_type = backend_spec.dtype_to_backend_type(input_x._attrs["dtype"])
    output_type = backend_spec.dtype_to_backend_type(output_y._attrs["dtype"])
    rank = len(input_x.shape())

    if input_type != output_type:
        raise NotImplementedError("input type must equal to output type")

    prefix = backend_spec.prefix

    return FUNC_TEMPLATE.render(
        header_files=header_files,
        elem_type=input_type,
        kernel=KERNEL_TEMPLATE.render(prefix=prefix, elem_type=input_type, rank=rank),
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"], prefix=prefix, rank=rank
        ),
    )


def gen_function_decl(func_attrs: Dict[str, Any], backend_spec) -> str:
    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"],
            prefix=backend_spec.prefix,
            rank=len(func_attrs["inputs"][0].shape()),
        ).strip()
    )
