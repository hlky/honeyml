#pragma once
#include <dinoml/device.h>
#include <dinoml/helpers.h>
#include <algorithm>
#include <cstdint>

namespace dinoml {

template <typename T>
__global__ void copy_first_frame_ncw_kernel(
    const T* __restrict__ in,
    T* __restrict__ out,
    int64_t N,
    int64_t C,
    int64_t W,
    int64_t WO) {
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = N * C;
  if (idx >= total)
    return;
  const int64_t n = idx / C;
  const int64_t c = idx - n * C;

  out[(n * C + c) * WO + 0] = LDG(&in[(n * C + c) * W + 0]);
}

template <typename T>
__global__ void avg_pool1d_k2s2_ncw_write_kernel(
    const T* __restrict__ in,
    T* __restrict__ out,
    int64_t N,
    int64_t C,
    int64_t W,
    int64_t WO_total,
    int64_t start_w,
    int64_t out_w_offset,
    int64_t WO_rest) {
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = N * C * WO_rest;
  if (idx >= total)
    return;

  int64_t t = idx;
  const int64_t ow = t % WO_rest;
  t /= WO_rest;
  const int64_t c = t % C;
  t /= C;
  const int64_t n = t;

  const int64_t w0 = start_w + 2 * ow;
  const int64_t w1 = w0 + 1;

  const int64_t in_base = (n * C + c) * W;
  const float a = (float)LDG(&in[in_base + w0]);
  const float b = (float)LDG(&in[in_base + w1]);

  const float y = 0.5f * (a + b);

  const int64_t out_base = (n * C + c) * WO_total;
  out[out_base + out_w_offset + ow] = (T)y;
}

template <typename T>
void avg_pool1d_compress_time_launch(
    const void* in_ptr,
    void* out_ptr,
    int64_t N,
    int64_t C,
    int64_t W,
    dinoml::DeviceStream stream) {
  const T* in = static_cast<const T*>(in_ptr);
  T* out = static_cast<T*>(out_ptr);

  const bool odd = (W & 1) && (W > 1);
  const int64_t WO = (W <= 1) ? 1 : (odd ? (1 + (W - 1) / 2) : (W / 2));

  constexpr int threads = 256;

  if (W <= 1) {
    const int64_t total = N * C;
    const int blocks = (int)((total + threads - 1) / threads);
    copy_first_frame_ncw_kernel<T>
        <<<blocks, threads, 0, stream>>>(in, out, N, C, W, /*WO*/ 1);
    return;
  }

  if (!odd) {
    const int64_t WO_rest = W / 2;
    const int64_t total = N * C * WO_rest;
    const int blocks = (int)((total + threads - 1) / threads);
    avg_pool1d_k2s2_ncw_write_kernel<T><<<blocks, threads, 0, stream>>>(
        in, out, N, C, W, WO, /*start_w*/ 0, /*out_off*/ 0, WO_rest);
    return;
  }

  {
    const int64_t total = N * C;
    const int blocks = (int)((total + threads - 1) / threads);
    copy_first_frame_ncw_kernel<T>
        <<<blocks, threads, 0, stream>>>(in, out, N, C, W, WO);
  }

  {
    const int64_t WO_rest = (W - 1) / 2;
    if (WO_rest > 0) {
      const int64_t total = N * C * WO_rest;
      const int blocks = (int)((total + threads - 1) / threads);
      avg_pool1d_k2s2_ncw_write_kernel<T><<<blocks, threads, 0, stream>>>(
          in, out, N, C, W, WO, /*start_w*/ 1, /*out_off*/ 1, WO_rest);
    }
  }
}

} // namespace dinoml
