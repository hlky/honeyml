#pragma once
#include <math.h>
#include <algorithm>
#include <cstdint>

#include "dinoml/device.h"

namespace dinoml {

template <typename T>
__device__ __forceinline__ T gelu_new(const T& x) {
  const float x3 = (float)(x * x * x);
  const T t = (T)tanhf((T)(0.79788456f * (float)(x + (T)(0.044715f * x3))));
  return ((T)0.5) * x * (((T)1.0) + t);
}

template <typename T>
__global__ void gelu_new_kernel(
    T* __restrict__ out,
    const T* __restrict__ in,
    const int last_dim) {
  const int64_t block_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < last_dim; idx += blockDim.x) {
    const T x = LDG(&in[block_idx * last_dim + idx]);
    out[block_idx * last_dim + idx] = gelu_new<T>(x);
  }
}

} // namespace dinoml

template <typename T>
void invoke_gelu_new(
    void* out,
    const void* in,
    int64_t n,
    int last_dim,
    dinoml::DeviceStream stream) {
  int64_t blocks = n / last_dim;
  int64_t threads = std::min(last_dim, 1024);

  dinoml::gelu_new_kernel<T><<<blocks, threads, 0, stream>>>(
      static_cast<T*>(out), static_cast<const T*>(in), last_dim);
}
