from typing import Any, Dict

from honey.backend import registry
from honey.backend.backend_spec import CUDASpec

import jinja2


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
struct AccumType { using type = float; };

template <typename T>
__device__ __forceinline__ float to_float(T v) {
    return (float)v;
}

template <>
__device__ __forceinline__ float to_float<half>(half v) {
    return __half2float(v);
}

#if defined(__CUDA_BF16_TYPES_EXIST__)
template <>
__device__ __forceinline__ float to_float<bfloat16>(bfloat16 v) {
    return __bfloat162float(v);
}
#endif

template <typename T>
__device__ __forceinline__ T from_float(float v);

template <>
__device__ __forceinline__ float from_float<float>(float v) { return v; }

template <>
__device__ __forceinline__ half from_float<half>(float v) { return __float2half(v); }

template <>
__device__ __forceinline__ bfloat16 from_float<bfloat16>(float v) { return __float2bfloat16(v); }
// -------- warp reduction helpers --------
__device__ __forceinline__ float warp_sum(float v) {
  // full mask
  unsigned mask = 0xffffffffu;
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(mask, v, offset);
  }
  return v;
}

__device__ __forceinline__ float block_sum(float v) {
  // Reduce within each warp
  v = warp_sum(v);

  // One float per warp in shared memory
  __shared__ float warp_sums[32]; // max 1024 threads => 32 warps
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;

  if (lane == 0) warp_sums[warp] = v;
  __syncthreads();

  // First warp reduces warp_sums
  float out = 0.0f;
  if (warp == 0) {
    int num_warps = (blockDim.x + 31) >> 5;
    out = (lane < num_warps) ? warp_sums[lane] : 0.0f;
    out = warp_sum(out);
  }
  return out; // valid in warp 0 lanes; broadcast below
}

// Broadcast a float from lane0 of warp0 to all threads
__device__ __forceinline__ float broadcast0(float v) {
  unsigned mask = 0xffffffffu;
  // lane 0 in warp 0 has v; others pass anything
  v = __shfl_sync(mask, v, 0);
  // now all lanes of warp0 have it; broadcast warp0 lane0 to all warps:
  __shared__ float s;
  if (threadIdx.x == 0) s = v;
  __syncthreads();
  return s;
}

__global__ void T5LayerNormKernelHalf2(
    half* __restrict__ out,
    const half* __restrict__ in,
    const half* __restrict__ w,
    int64_t M, int64_t N,
    float eps)
{
  int row = (int)blockIdx.x;
  if (row >= M) return;

  int64_t N2 = N >> 1; // N even required
  const half2* in2 = reinterpret_cast<const half2*>(in + row * N);
  const half2* w2  = reinterpret_cast<const half2*>(w);
  half2* out2      = reinterpret_cast<half2*>(out + row * N);

  float sumsq = 0.0f;
  for (int i = threadIdx.x; i < N2; i += blockDim.x) {
    half2 x2 = in2[i];
    float2 xf = __half22float2(x2);
    sumsq += xf.x * xf.x + xf.y * xf.y;
  }

  float total = block_sum(sumsq);
  float inv_rms = 0.0f;
  if (threadIdx.x == 0) {
      inv_rms = rsqrtf(total / (float)N + eps);
  }
  inv_rms = broadcast0(inv_rms);

  for (int i = threadIdx.x; i < N2; i += blockDim.x) {
    half2 x2 = in2[i];
    half2 g2 = w2[i];
    float2 xf = __half22float2(x2);
    float2 gf = __half22float2(g2);

    float2 yf;
    yf.x = xf.x * inv_rms * gf.x;
    yf.y = xf.y * inv_rms * gf.y;

    out2[i] = __floats2half2_rn(yf.x, yf.y);
  }
}

__global__ void T5LayerNormKernelBf162(
    bfloat16* __restrict__ out,
    const bfloat16* __restrict__ in,
    const bfloat16* __restrict__ w,
    int64_t M, int64_t N,
    float eps)
{
  int row = (int)blockIdx.x;
  if (row >= M) return;

  int64_t N2 = N >> 1;
  const bfloat162* in2 = reinterpret_cast<const bfloat162*>(in + row * N);
  const bfloat162* w2  = reinterpret_cast<const bfloat162*>(w);
  bfloat162* out2      = reinterpret_cast<bfloat162*>(out + row * N);

  float sumsq = 0.0f;
  for (int i = threadIdx.x; i < N2; i += blockDim.x) {
    bfloat162 x2 = in2[i];
    // unpack
    float x0 = __bfloat162float(x2.x);
    float x1 = __bfloat162float(x2.y);
    sumsq += x0 * x0 + x1 * x1;
  }

  float total = block_sum(sumsq);
  float inv_rms = 0.0f;
  if (threadIdx.x == 0) {
      inv_rms = rsqrtf(total / (float)N + eps);
  }
  inv_rms = broadcast0(inv_rms);

  for (int i = threadIdx.x; i < N2; i += blockDim.x) {
    bfloat162 x2 = in2[i];
    bfloat162 g2 = w2[i];
    float x0 = __bfloat162float(x2.x);
    float x1 = __bfloat162float(x2.y);
    float g0 = __bfloat162float(g2.x);
    float g1 = __bfloat162float(g2.y);

    bfloat162 y2;
    y2.x = __float2bfloat16(x0 * inv_rms * g0);
    y2.y = __float2bfloat16(x1 * inv_rms * g1);
    out2[i] = y2;
  }
}

template <typename T>
__global__ void T5LayerNormKernelScalar(
    T* __restrict__ out,
    const T* __restrict__ in,
    const T* __restrict__ w,
    int64_t M, int64_t N,
    float eps)
{
  int row = (int)blockIdx.x;
  if (row >= M) return;

  float sumsq = 0.0f;
  for (int col = threadIdx.x; col < N; col += blockDim.x) {
    float x = to_float<T>(in[row * N + col]);
    sumsq += x * x;
  }

  float total = block_sum(sumsq);
  float inv_rms = 0.0f;
  if (threadIdx.x == 0) {
      inv_rms = rsqrtf(total / (float)N + eps);
  }
  inv_rms = broadcast0(inv_rms);

  for (int col = threadIdx.x; col < N; col += blockDim.x) {
    float x = to_float<T>(in[row * N + col]);
    float g = to_float<T>(w[col]);
    out[row * N + col] = from_float<T>(x * inv_rms * g);
  }
}

void invoke_t5_layer_norm(
    half* out, const half* in, const half* w,
    int64_t M, int64_t N, float eps, cudaStream_t stream)
{
  int threads = 256; // often a good default for LN/RMSNorm
  if (threads > 1024) threads = 1024;

  dim3 grid((unsigned)M);
  dim3 block((unsigned)threads);

  bool even = (N % 2) == 0;
  bool aligned =
      (((uintptr_t)out % 4) == 0) &&
      (((uintptr_t)in  % 4) == 0) &&
      (((uintptr_t)w   % 4) == 0);

  if (even && aligned) {
    T5LayerNormKernelHalf2<<<grid, block, 0, stream>>>(
        out, in, w, M, N, eps);
  } else {
    T5LayerNormKernelScalar<half><<<grid, block, 0, stream>>>(
        out, in, w, M, N, eps);
  }
}

void invoke_t5_layer_norm(
    bfloat16* out, const bfloat16* in, const bfloat16* w,
    int64_t M, int64_t N, float eps, cudaStream_t stream)
{
  int threads = 256;
  dim3 grid((unsigned)M);
  dim3 block((unsigned)threads);

  bool even = (N % 2) == 0;
  bool aligned =
      (((uintptr_t)out % 4) == 0) &&
      (((uintptr_t)in  % 4) == 0) &&
      (((uintptr_t)w   % 4) == 0);

  if (even && aligned) {
    T5LayerNormKernelBf162<<<grid, block, 0, stream>>>(
        out, in, w, M, N, eps);
  } else {
    T5LayerNormKernelScalar<bfloat16><<<grid, block, 0, stream>>>(
        out, in, w, M, N, eps);
  }
}

void invoke_t5_layer_norm(
    float* out, const float* in, const float* w,
    int64_t M, int64_t N, float eps, cudaStream_t stream)
{
  int threads = 256;
  dim3 grid((unsigned)M);
  dim3 block((unsigned)threads);

  T5LayerNormKernelScalar<float><<<grid, block, 0, stream>>>(
      out, in, w, M, N, eps);
}
"""
)

FUNC_TEMPLATE = jinja2.Template(
    r"""
{{header_files}}

namespace {

{{kernel}}

}  // namespace

{{func_signature}}
{
    invoke_t5_layer_norm(
        static_cast<{{elem_type}}*>(output),
        static_cast<const {{elem_type}}*>(input),
        static_cast<const {{elem_type}}*>(weight),
        M,
        N,
        eps,
        stream);
}
"""
)

FUNC_SIGNATURE = jinja2.Template(
    r"""
void {{func_name}}(void* output,
                   const void* input,
                   const void* weight,
                   int64_t M,
                   int64_t N,
                   float eps,
                   {{prefix}}Stream_t stream)
"""
)

FUNC_DECL = jinja2.Template(
    r"""
    {{func_signature}};
"""
)

FUNC_CALL_TEMPLATE = jinja2.Template(
    r"""
{{indent}}{{func_name}}(
{{indent}}   {{output}}, {{input}}, {{weight}}, {{M}}, {{N}}, {{eps}}, stream /* default stream */
{{indent}});
"""
)


def _render_dim(dim) -> str:
    name = dim._attrs.get("name", None)
    if name is not None:
        return name
    return str(dim.symbolic_value())


def gen_function_call(func_attrs: Dict[str, Any], indent="  ") -> str:
    out = func_attrs["outputs"][0]
    x = func_attrs["inputs"][0]
    w = func_attrs["inputs"][1]

    out_name = out._attrs["name"]
    x_name = x._attrs["name"]
    w_name = w._attrs["name"]

    x_shape = x.shape()
    N = _render_dim(x_shape[-1])
    M = " * ".join([_render_dim(d) for d in x_shape[:-1]])

    eps = func_attrs["eps"]
    eps_value = eps if isinstance(eps, (int, float)) else str(eps)

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=out_name,
        input=x_name,
        weight=w_name,
        M=M,
        N=N,
        eps=eps_value,
        indent=indent,
    )


def gen_function(func_attrs: Dict[str, Any], header_files: str, backend_spec) -> str:
    y = func_attrs["outputs"][0]
    elem_type = backend_spec.dtype_to_backend_type(y._attrs["dtype"])

    if y._attrs["dtype"] not in ["float32", "float16", "bfloat16"]:
        raise NotImplementedError("Unsupported dtype for t5_layer_norm: " + elem_type)

    prefix = backend_spec.prefix

    return FUNC_TEMPLATE.render(
        header_files=header_files,
        elem_type=elem_type,
        kernel=KERNEL_TEMPLATE.render(
            prefix=prefix,
            elem_type=elem_type,
        ),
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"],
            prefix=prefix,
        ),
    )


def gen_function_decl(func_attrs: Dict[str, Any], backend_spec) -> str:
    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"],
            prefix=backend_spec.prefix,
        ).strip()
    )


@registry.reg("cuda.t5_layer_norm.gen_function")
def cuda_t5_layer_norm_gen_function(func_attrs: Dict[str, Any]) -> str:
    return gen_function(func_attrs, CUDA_HEADER_FILES, CUDASpec())


@registry.reg("cuda.t5_layer_norm.func_decl")
def cuda_t5_layer_norm_gen_function_decl(func_attrs: Dict[str, Any]) -> str:
    return gen_function_decl(func_attrs, CUDASpec())


@registry.reg("cuda.t5_layer_norm.func_call")
def cuda_t5_layer_norm_gen_function_call(
    func_attrs: Dict[str, Any], indent="  "
) -> str:
    return gen_function_call(func_attrs, indent)
