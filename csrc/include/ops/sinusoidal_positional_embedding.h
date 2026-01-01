#pragma once
#include <math.h>
#include <algorithm>
#include <cstdint>

#include "dinoml/device.h"

namespace dinoml {

template <typename T>
__global__ void sinusoidal_positional_embedding_kernel(
    T* __restrict__ out,
    const T* __restrict__ x,
    int batch,
    int seq_len,
    int embed_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * seq_len * embed_dim;
  if (idx >= total) return;

  int d = idx % embed_dim;
  int tmp = idx / embed_dim;
  int pos = tmp % seq_len;

  // match torch:
  // div_term[k] = exp((2k) * (-log(10000)/embed_dim))
  // even d=2k uses sin(pos*div_term[k])
  // odd  d=2k+1 uses cos(pos*div_term[k])
  const int k = d >> 1;                 // k = floor(d/2)
  const float two_k = (float)(k << 1);  // 2k

  const float inv = -logf(10000.0f) / (float)embed_dim;
  const float div_term = expf(inv * two_k);
  const float angle = (float)pos * div_term;

  const float pe = (d & 1) ? cosf(angle) : sinf(angle);

  out[idx] = (T)((float)LDG(&x[idx]) + pe);
}

template <typename T>
void invoke_sinusoidal_positional_embedding(
    void* out,
    const void* x,
    int batch,
    int seq_len,
    int embed_dim,
    dinoml::DeviceStream stream) {
  int total = batch * seq_len * embed_dim;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;

  sinusoidal_positional_embedding_kernel<T><<<blocks, threads, 0, stream>>>(
      (T*)out, (const T*)x, batch, seq_len, embed_dim);
}

} // namespace dinoml
