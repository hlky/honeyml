#pragma once

#include <cstdint>
#include "dinoml/device.h"

namespace dinoml {

template <typename T>
__global__ void flip_kernel(
    T* __restrict__ out,
    const T* __restrict__ in,
    int64_t n,
    const int64_t* __restrict__ sizes,
    const int64_t* __restrict__ flip_dims,
    int ndims,
    int nflip) {
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += blockDim.x * gridDim.x) {
    int64_t tmp = idx;
    int64_t coords[16];

#pragma unroll
    for (int i = ndims - 1; i >= 0; --i) {
      coords[i] = tmp % sizes[i];
      tmp /= sizes[i];
    }

    // apply flips
    for (int j = 0; j < nflip; ++j) {
      int d = flip_dims[j];
      coords[d] = sizes[d] - 1 - coords[d];
    }

    // recompute linear index
    int64_t in_idx = 0;
    for (int i = 0; i < ndims; ++i) {
      in_idx = in_idx * sizes[i] + coords[i];
    }

    out[idx] = LDG(&in[in_idx]);
  }
}

} // namespace dinoml

template <typename T>
inline void invoke_flip(
    void* out,
    const void* in,
    int64_t n,
    const int64_t* sizes,
    const int64_t* flip_dims,
    int ndims,
    int nflip,
    dinoml::DeviceStream stream) {
  constexpr int threads = 256;
  int blocks = (n + threads - 1) / threads;
  if (blocks > 65535)
    blocks = 65535;

  dinoml::flip_kernel<T><<<blocks, threads, 0, stream>>>(
      static_cast<T*>(out),
      static_cast<const T*>(in),
      n,
      sizes,
      flip_dims,
      ndims,
      nflip);
}
