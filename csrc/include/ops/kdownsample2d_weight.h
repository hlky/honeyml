#pragma once
#include <dinoml/device.h>

namespace dinoml {

template <typename T>
__global__ void kdownsample2d_weight_kernel(T* out, int channels) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = channels * channels * 4 * 4;
  if (idx >= total)
    return;

  int tmp = idx;
  int kw = tmp % 4;
  tmp /= 4;
  int kh = tmp % 4;
  tmp /= 4;
  int c_in = tmp % channels;
  int c_out = tmp / channels;

  // Only diagonal channels get kernel
  if (c_in != c_out) {
    out[idx] = (T)0;
    return;
  }

  const float k1d[4] = {0.125f, 0.375f, 0.375f, 0.125f};
  out[idx] = (T)(k1d[kh] * k1d[kw]);
}

} // namespace dinoml

template <typename T>
void invoke_kdownsample2d_weight(
    void* out,
    int channels,
    dinoml::DeviceStream stream) {
  int total = channels * channels * 4 * 4;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;

  dinoml::kdownsample2d_weight_kernel<T>
      <<<blocks, threads, 0, stream>>>(static_cast<T*>(out), channels);
}
