#pragma once
#include <math.h>
#include <algorithm>
#include <cstdint>

#include "dinoml/device.h"

namespace dinoml {

#ifndef DINOML_PI_F
#define DINOML_PI_F 3.141592653589793f
#endif

// Computes:
// if log: x = log(x)
// x_proj = x * weight[j] * 2*pi
// out = [sin(x_proj), cos(x_proj)] or flipped [cos, sin]
template <typename T>
__global__ void gaussian_fourier_projection_kernel(
    T* __restrict__ out, // [N, 2*E]
    const T* __restrict__ x, // [N]
    const T* __restrict__ weight, // [E]
    const int64_t n, // N
    const int embedding_size, // E
    const int do_log, // 0/1
    const int flip_sin_to_cos) { // 0/1
  const int64_t row = (int64_t)blockIdx.x; // over N
  if (row >= n) {
    return;
  }

  // Load x and optionally log
  float xf = (float)LDG(&x[row]);
  if (do_log) {
    xf = logf(xf);
  }

  const int out_cols = embedding_size * 2;

  // Each thread produces some columns (2*E)
  for (int c = (int)threadIdx.x; c < out_cols; c += (int)blockDim.x) {
    // Map column to weight index and which function
    // layout without flip: [sin(0..E-1), cos(0..E-1)]
    // layout with flip:    [cos(0..E-1), sin(0..E-1)]
    const int j = c % embedding_size;
    const int in_first_half = (c < embedding_size) ? 1 : 0;

    const float wf = (float)LDG(&weight[j]);
    const float ang = xf * wf * 2.0f * DINOML_PI_F;

    float val = 0.0f;
    if (!flip_sin_to_cos) {
      // first half sin, second half cos
      val = in_first_half ? sinf(ang) : cosf(ang);
    } else {
      // first half cos, second half sin
      val = in_first_half ? cosf(ang) : sinf(ang);
    }

    out[row * (int64_t)out_cols + (int64_t)c] = (T)val;
  }
}

template <typename T>
void invoke_gaussian_fourier_projection(
    void* out,
    const void* x,
    const void* weight,
    int64_t n,
    int embedding_size,
    int do_log,
    int flip_sin_to_cos,
    dinoml::DeviceStream stream) {
  const int threads = std::min(embedding_size * 2, 1024);
  dinoml::gaussian_fourier_projection_kernel<T><<<n, threads, 0, stream>>>(
      static_cast<T*>(out),
      static_cast<const T*>(x),
      static_cast<const T*>(weight),
      n,
      embedding_size,
      do_log,
      flip_sin_to_cos);
}

} // namespace dinoml
