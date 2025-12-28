#pragma once
#include <math.h>
#include <cstdint>
#include <type_traits>

#include <curand_kernel.h>

#include "dinoml/device.h"

namespace dinoml {

template <typename T>
__device__ __forceinline__ float to_float(T v) {
  if constexpr (std::is_same_v<T, float>)
    return v;
  else if constexpr (std::is_same_v<T, half>)
    return __half2float(v);
  else if constexpr (std::is_same_v<T, bfloat16>)
    return __bfloat162float(v);
}

template <typename T>
__device__ __forceinline__ T from_float(float v) {
  if constexpr (std::is_same_v<T, float>)
    return v;
  else if constexpr (std::is_same_v<T, half>)
    return __float2half(v);
  else if constexpr (std::is_same_v<T, bfloat16>)
    return __float2bfloat16(v);
}

template <typename T>
__global__ void randn_philox(
    T* out,
    int64_t n,
    float mean,
    float std,
    uint64_t seed,
    uint64_t philox_offset) {
  const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t stride = (int64_t)blockDim.x * gridDim.x;

  curandStatePhilox4_32_10_t state;
  curand_init(
      (unsigned long long)seed,
      (unsigned long long)tid,
      (unsigned long long)philox_offset,
      &state);

  int64_t idx = tid;
  while (idx < n) {
    float4 r = curand_normal4(&state);

    if (idx < n)
      out[idx] = from_float<T>(r.x * std + mean);
    int64_t i1 = idx + stride;
    if (i1 < n)
      out[i1] = from_float<T>(r.y * std + mean);
    int64_t i2 = idx + 2 * stride;
    if (i2 < n)
      out[i2] = from_float<T>(r.z * std + mean);
    int64_t i3 = idx + 3 * stride;
    if (i3 < n)
      out[i3] = from_float<T>(r.w * std + mean);

    idx += stride * 4;
  }
}

} // namespace dinoml

template <typename ElemOutputType>
void invoke_randn(
    void* out,
    int64_t n,
    float mean,
    float std,
    uint64_t seed,
    uint64_t offset_groups,
    dinoml::DeviceStream stream) {
  static_assert(
      std::is_same_v<ElemOutputType, float> ||
          std::is_same_v<ElemOutputType, half> ||
          std::is_same_v<ElemOutputType, bfloat16>,
      "Unsupported dtype for invoke_randn");

  auto [counter_offset, grid, block] =
      calc_execution_policy(n, /*unroll_factor=*/4);
  dinoml::randn_philox<ElemOutputType><<<grid, block, 0, stream>>>(
      static_cast<ElemOutputType*>(out), n, mean, std, seed, offset_groups);
}
