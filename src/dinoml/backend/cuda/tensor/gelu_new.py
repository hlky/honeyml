from typing import Any, Dict, List

from dinoml.backend import registry
from dinoml.backend.backend_spec import CUDASpec

import jinja2

from dinoml.compiler.base import IntImm, IntVar


CUDA_HEADER_FILES = """
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

using bfloat16 = __nv_bfloat16;
using bfloat162 = __nv_bfloat162;
"""


KERNEL_TEMPLATE = jinja2.Template(
    """
#include <stdint.h>

template <typename T>
__device__ __forceinline__ float to_float(T v) { return (float)v; }

template <>
__device__ __forceinline__ float to_float<half>(half v) {
    return __half2float(v);
}

template <>
__device__ __forceinline__ float to_float<bfloat16>(bfloat16 v) {
    return __bfloat162float(v);
}

template <typename T>
__device__ __forceinline__ T from_float(float v);

template <>
__device__ __forceinline__ float from_float<float>(float v) { return v; }

template <>
__device__ __forceinline__ half from_float<half>(float v) {
    return __float2half(v);
}

template <>
__device__ __forceinline__ bfloat16 from_float<bfloat16>(float v) {
    return __float2bfloat16(v);
}

__device__ __forceinline__ float gelu_new(float x) {
    const float kAlpha = 0.7978845608028654f; // sqrt(2/pi)
    const float kBeta  = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(kAlpha * (x + kBeta * x3)));
}

template <typename T>
__global__ void GeluNewKernel(
    T* __restrict__ out,
    const T* __restrict__ in,
    int64_t n)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = to_float<T>(in[idx]);
    out[idx] = from_float<T>(gelu_new(x));
}

__global__ void GeluNewKernelHalf2(
    half* __restrict__ out,
    const half* __restrict__ in,
    int64_t n)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t i = idx * 2;
    if (i + 1 >= n) return;

    half2 x2 = reinterpret_cast<const half2*>(in)[idx];
    float2 xf = __half22float2(x2);

    xf.x = gelu_new(xf.x);
    xf.y = gelu_new(xf.y);

    reinterpret_cast<half2*>(out)[idx] =
        __floats2half2_rn(xf.x, xf.y);
}

__global__ void GeluNewKernelBf162(
    bfloat16* __restrict__ out,
    const bfloat16* __restrict__ in,
    int64_t n)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t i = idx * 2;
    if (i + 1 >= n) return;

    bfloat162 x2 = reinterpret_cast<const bfloat162*>(in)[idx];

    float x0 = __bfloat162float(x2.x);
    float x1 = __bfloat162float(x2.y);

    x0 = gelu_new(x0);
    x1 = gelu_new(x1);

    reinterpret_cast<bfloat162*>(out)[idx] =
        bfloat162{__float2bfloat16(x0), __float2bfloat16(x1)};
}

template <typename T>
void invoke_gelu_new(
    T* out,
    const T* in,
    int64_t n,
    {{prefix}}Stream_t stream)
{
    if (n <= 0) return;

    int threads = 256;

    if constexpr (std::is_same<T, half>::value) {
        int64_t n2 = n >> 1;
        int blocks = (int)((n2 + threads - 1) / threads);
        GeluNewKernelHalf2<<<blocks, threads, 0, stream>>>(out, in, n);
    } else if constexpr (std::is_same<T, bfloat16>::value) {
        int64_t n2 = n >> 1;
        int blocks = (int)((n2 + threads - 1) / threads);
        GeluNewKernelBf162<<<blocks, threads, 0, stream>>>(out, in, n);
    } else {
        int blocks = (int)((n + threads - 1) / threads);
        GeluNewKernel<T><<<blocks, threads, 0, stream>>>(out, in, n);
    }
}
"""
)


FUNC_DECL = jinja2.Template(
    r"""
    {{func_signature}};
"""
)

FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(void* output,
                   const void* input,
                   int64_t numel,
                   {{prefix}}Stream_t stream)
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
    invoke_gelu_new<{{elem_type}}>(
        static_cast<{{elem_type}}*>(output),
        static_cast<const {{elem_type}}*>(input),
        numel,
        stream);
}
"""
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    """
{{indent}}{{func_name}}(
{{indent}}   {{output}}, {{input}}, {{numel}}, stream
{{indent}});
"""
)


def gen_int_var_product_str(
    int_vars: List[IntVar],
) -> str:
    res = []
    for int_var in int_vars:
        if isinstance(int_var, IntImm):
            res.append(str(int_var._attrs["values"][0]))
        elif isinstance(int_var, IntVar):
            res.append(int_var._attrs["name"])
        else:
            raise RuntimeError(
                "A dim must be an IntVar! Current type: {}".format(type(int_var))
            )

    return " * ".join(res) if res else "1"


def gen_function_call(func_attrs: Dict[str, Any], indent="  ") -> str:
    x = func_attrs["inputs"][0]
    y = func_attrs["outputs"][0]

    numel = gen_int_var_product_str(x._attrs["shape"])

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=y._attrs["name"],
        input=x._attrs["name"],
        numel=numel,
        indent=indent,
    )


def gen_function(func_attrs: Dict[str, Any], header_files: str, backend_spec) -> str:
    y = func_attrs["outputs"][0]
    elem_type = backend_spec.dtype_to_backend_type(y._attrs["dtype"])

    return FUNC_TEMPLATE.render(
        header_files=header_files,
        elem_type=elem_type,
        kernel=KERNEL_TEMPLATE.render(
            elem_type=elem_type,
            prefix=backend_spec.prefix,
        ),
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"],
            prefix=backend_spec.prefix,
        ),
    )


def gen_function_decl(func_attrs: Dict[str, Any], backend_spec) -> str:
    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"],
            prefix=backend_spec.prefix,
        ).strip()
    )


@registry.reg("cuda.gelu_new.gen_function")
def cuda_gelu_new_gen_function(func_attrs):
    return gen_function(func_attrs, CUDA_HEADER_FILES, CUDASpec())


@registry.reg("cuda.gelu_new.func_decl")
def cuda_gelu_new_gen_function_decl(func_attrs: Dict[str, Any]) -> str:
    return gen_function_decl(func_attrs, CUDASpec())


@registry.reg("cuda.gelu_new.func_call")
def cuda_gelu_new_func_call(func_attrs, indent="  "):
    return gen_function_call(func_attrs, indent)
