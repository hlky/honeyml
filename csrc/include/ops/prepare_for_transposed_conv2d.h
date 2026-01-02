#pragma once
#include <cstdint>
#include "dinoml/device.h"

namespace dinoml {

template <typename T>
__global__ void prepare_for_transposed_conv2d_kernel(
    T* __restrict__ out,
    const T* __restrict__ in,
    int64_t N,
    int64_t H,
    int64_t W,
    int64_t C,
    int64_t stride_h,
    int64_t stride_w) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = N * H * W * C;
  if (idx >= total)
    return;

  int64_t c = idx % C;
  int64_t tmp = idx / C;
  int64_t w = tmp % W;
  tmp /= W;
  int64_t h = tmp % H;
  int64_t n = tmp / H;

  int64_t out_h = h * stride_h;
  int64_t out_w = w * stride_w;

  int64_t out_W = (W - 1) * stride_w + 1;
  int64_t out_H = (H - 1) * stride_h + 1;

  int64_t out_idx = ((n * out_H + out_h) * out_W + out_w) * C + c;

  out[out_idx] = in[idx];
}

template <typename T>
inline void invoke_prepare_for_transposed_conv2d(
    void* out,
    const void* in,
    int64_t N,
    int64_t H,
    int64_t W,
    int64_t C,
    int64_t stride_h,
    int64_t stride_w,
    dinoml::DeviceStream stream) {
  int64_t total = N * H * W * C;
  int threads = 1024;
  int blocks = (total + threads - 1) / threads;

  prepare_for_transposed_conv2d_kernel<T><<<blocks, threads, 0, stream>>>(
      static_cast<T*>(out),
      static_cast<const T*>(in),
      N,
      H,
      W,
      C,
      stride_h,
      stride_w);
}

} // namespace dinoml
