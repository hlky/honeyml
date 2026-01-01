#pragma once
#include <math.h>
#include <algorithm>
#include <cstdint>

#include "dinoml/device.h"

namespace dinoml {

template <typename T>
__global__ void fourier_embeds_from_boundingbox_kernel(
    T* __restrict__ out,
    const T* __restrict__ box,
    int embed_dim,
    int batch,
    int num_boxes) {
  const int idx = blockIdx.x;
  if (idx >= batch * num_boxes)
    return;

  const int b = idx / num_boxes;
  const int n = idx % num_boxes;

  const int base_out = idx * (embed_dim * 8);
  const int base_box = idx * 4;

  for (int k = threadIdx.x; k < embed_dim; k += blockDim.x) {
    const float scale = powf(100.0f, (float)k / (float)embed_dim);

    for (int c = 0; c < 4; ++c) {
      float v = box[base_box + c] * (T)scale;
      float s = sinf(v);
      float c_ = cosf(v);

      int base = base_out + k * 8 + c;
      out[base + 0 * 4] = (T)s;
      out[base + 1 * 4] = (T)c_;
    }
  }
}

template <typename T>
void invoke_get_fourier_embeds_from_boundingbox(
    void* out,
    const void* box,
    int embed_dim,
    int batch,
    int num_boxes,
    dinoml::DeviceStream stream) {
  int blocks = batch * num_boxes;
  int threads = 256;

  fourier_embeds_from_boundingbox_kernel<T><<<blocks, threads, 0, stream>>>(
      (T*)out, (const T*)box, embed_dim, batch, num_boxes);
}

} // namespace dinoml
