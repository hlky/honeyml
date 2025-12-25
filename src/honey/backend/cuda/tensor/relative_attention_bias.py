from typing import Any, Dict
import jinja2

from honey.backend import registry
from honey.backend.backend_spec import CUDASpec


KERNEL_TEMPLATE = jinja2.Template(
    """
#include <stdint.h>
#include <math.h>

template <typename T>
__device__ __forceinline__ float to_float(T v) { return (float)v; }

template <>
__device__ __forceinline__ float to_float<half>(half v) { return __half2float(v); }

template <>
__device__ __forceinline__ float to_float<bfloat16>(bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ __forceinline__ T from_float(float v);

template <>
__device__ __forceinline__ float from_float<float>(float v) { return v; }

template <>
__device__ __forceinline__ half from_float<half>(float v) { return __float2half(v); }

template <>
__device__ __forceinline__ bfloat16 from_float<bfloat16>(float v) { return __float2bfloat16(v); }

__device__ __forceinline__ int relpos_bucket(
    int rel, int bidirectional, int num_buckets, int max_distance)
{
    int bucket = 0;
    int n = num_buckets;

    if (bidirectional != 0) {
        n = n / 2;
        if (rel > 0) bucket += n;
        rel = rel > 0 ? rel : -rel;
    } else {
        // rel = -min(rel, 0)
        rel = rel < 0 ? -rel : 0;
    }

    // now rel in [0, inf)
    int max_exact = n / 2;

    if (rel < max_exact) {
        bucket += rel;
        return bucket;
    }

    // large bucket: max_exact + log(rel/max_exact)/log(max_distance/max_exact) * (n-max_exact)
    float rel_f = (float)rel;
    float max_exact_f = (float)max_exact;
    float max_dist_f = (float)max_distance;

    float log_ratio = logf(rel_f / max_exact_f);
    float log_base  = logf(max_dist_f / max_exact_f);

    float scaled = (log_ratio / log_base) * (float)(n - max_exact);
    int b = max_exact + (int)scaled;

    if (b > (n - 1)) b = (n - 1);
    bucket += b;
    return bucket;
}

// output: [1, H, Q, K] flattened as (((h*Q)+q)*K + k)
template <typename T>
__global__ void RelAttnBiasKernel(
    T* __restrict__ out,
    const T* __restrict__ emb,   // [num_buckets, H] row-major: emb[bucket*H + h]
    int Q,
    int K,
    int H,
    int num_buckets,
    int max_distance,
    int bidirectional)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)H * (int64_t)Q * (int64_t)K;
    if (idx >= total) return;

    int64_t t = idx;
    int k = (int)(t % K); t /= K;
    int q = (int)(t % Q); t /= Q;
    int h = (int)t;

    int rel = k - q;
    int b = relpos_bucket(rel, bidirectional, num_buckets, max_distance);

    // gather embedding and write
    float v = to_float<T>(emb[b * H + h]);
    out[idx] = from_float<T>(v);
}

void invoke_relative_attention_bias(
    {{elem_type}}* out,
    const {{elem_type}}* emb,
    int Q,
    int K,
    int H,
    int num_buckets,
    int max_distance,
    int bidirectional,
    {{prefix}}Stream_t stream)
{
    int threads = 256;
    int64_t total = (int64_t)H * (int64_t)Q * (int64_t)K;
    int blocks = (int)((total + threads - 1) / threads);

    RelAttnBiasKernel<{{elem_type}}><<<blocks, threads, 0, stream>>>(
        out, emb, Q, K, H, num_buckets, max_distance, bidirectional);
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
    invoke_relative_attention_bias(
        static_cast<{{elem_type}}*>(output),
        static_cast<const {{elem_type}}*>(embedding),
        Q,
        K,
        H,
        num_buckets,
        max_distance,
        bidirectional,
        stream);
}
"""
)

FUNC_SIGNATURE = jinja2.Template(
    """
void {{func_name}}(void* output,
                   const void* embedding,
                   int Q,
                   int K,
                   int H,
                   int num_buckets,
                   int max_distance,
                   int bidirectional,
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
{{indent}}   {{output}}, {{embedding}}, {{Q}}, {{K}}, {{H}}, {{num_buckets}}, {{max_distance}}, {{bidirectional}}, stream /* default stream */
{{indent}});
"""
)


def gen_function_call(func_attrs: Dict[str, Any], indent="  ") -> str:
    out = func_attrs["outputs"][0]
    emb = func_attrs["inputs"][0]

    out_name = out._attrs["name"]
    emb_name = emb._attrs["name"]

    # output is [1, H, Q, K]
    H = out.shape()[1]
    Q = out.shape()[2]
    K = out.shape()[3]

    def dim_expr(d):
        n = d._attrs.get("name", None)
        return n if n is not None else str(d.symbolic_value())

    H_expr = dim_expr(H)
    Q_expr = dim_expr(Q)
    K_expr = dim_expr(K)

    return FUNC_CALL_TEMPLATE.render(
        func_name=func_attrs["name"],
        output=out_name,
        embedding=emb_name,
        Q=Q_expr,
        K=K_expr,
        H=H_expr,
        num_buckets=int(func_attrs["num_buckets"]),
        max_distance=int(func_attrs["max_distance"]),
        bidirectional="1" if bool(func_attrs["bidirectional"]) else "0",
        indent=indent,
    )


def gen_function(func_attrs: Dict[str, Any], header_files: str, backend_spec) -> str:
    y = func_attrs["outputs"][0]
    elem_type = backend_spec.dtype_to_backend_type(y._attrs["dtype"])

    if y._attrs["dtype"] not in ["float32", "float16", "bfloat16"]:
        raise NotImplementedError(
            "Unsupported dtype for relative_attention_bias: " + elem_type
        )

    return FUNC_TEMPLATE.render(
        header_files=header_files,
        elem_type=elem_type,
        kernel=KERNEL_TEMPLATE.render(prefix=backend_spec.prefix, elem_type=elem_type),
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"], prefix=backend_spec.prefix
        ),
    )


def gen_function_decl(func_attrs: Dict[str, Any], backend_spec) -> str:
    return FUNC_DECL.render(
        func_signature=FUNC_SIGNATURE.render(
            func_name=func_attrs["name"], prefix=backend_spec.prefix
        ).strip()
    )


CUDA_HEADER_FILES = """
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

using bfloat16 = __nv_bfloat16;
using bfloat162 = __nv_bfloat162;
"""


@registry.reg("cuda.relative_attention_bias.gen_function")
def cuda_rel_attn_bias_gen_function(func_attrs: Dict[str, Any]) -> str:
    return gen_function(func_attrs, CUDA_HEADER_FILES, CUDASpec())


@registry.reg("cuda.relative_attention_bias.func_decl")
def cuda_rel_attn_bias_gen_function_decl(func_attrs: Dict[str, Any]) -> str:
    return gen_function_decl(func_attrs, CUDASpec())


@registry.reg("cuda.relative_attention_bias.func_call")
def cuda_rel_attn_bias_gen_function_call(
    func_attrs: Dict[str, Any], indent="  "
) -> str:
    return gen_function_call(func_attrs, indent)
