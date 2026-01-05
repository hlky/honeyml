#pragma once
#include <cstdint>
#include "dinoml/device.h"

namespace dinoml {

template <typename T, int MAX_RANK = 16>
__global__ void flip_kernel(
    T* __restrict__ out,
    const T* __restrict__ in,
    int64_t n,
    int ndims,
    int nflip,
    int64_t s0,
    int64_t s1,
    int64_t s2,
    int64_t s3,
    int64_t s4,
    int64_t s5,
    int64_t s6,
    int64_t s7,
    int64_t s8,
    int64_t s9,
    int64_t s10,
    int64_t s11,
    int64_t s12,
    int64_t s13,
    int64_t s14,
    int64_t s15,
    int f0,
    int f1,
    int f2,
    int f3,
    int f4,
    int f5,
    int f6,
    int f7) {
  int64_t sizes[MAX_RANK] = {
      s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15};

  int flips[8] = {f0, f1, f2, f3, f4, f5, f6, f7};

  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += blockDim.x * gridDim.x) {
    int64_t coords[16];
    int64_t tmp = idx;

#pragma unroll
    for (int i = ndims - 1; i >= 0; --i) {
      coords[i] = tmp % sizes[i];
      tmp /= sizes[i];
    }

#pragma unroll
    for (int j = 0; j < nflip; ++j) {
      int d = flips[j];
      coords[d] = sizes[d] - 1 - coords[d];
    }

    int64_t in_idx = 0;
    for (int i = 0; i < ndims; ++i)
      in_idx = in_idx * sizes[i] + coords[i];

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

  dinoml::flip_kernel<T><<<blocks, threads, 0, stream>>>(
      static_cast<T*>(out),
      static_cast<const T*>(in),
      n,
      ndims,
      nflip,
      sizes[0],
      sizes[1],
      sizes[2],
      sizes[3],
      sizes[4],
      sizes[5],
      sizes[6],
      sizes[7],
      sizes[8],
      sizes[9],
      sizes[10],
      sizes[11],
      sizes[12],
      sizes[13],
      sizes[14],
      sizes[15],
      flip_dims[0],
      flip_dims[1],
      flip_dims[2],
      flip_dims[3],
      flip_dims[4],
      flip_dims[5],
      flip_dims[6],
      flip_dims[7]);
}
