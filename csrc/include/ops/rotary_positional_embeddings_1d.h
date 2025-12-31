#pragma once
#include <math.h>
#include <algorithm>
#include <cstdint>

#include "dinoml/device.h"

namespace dinoml {

__device__ __forceinline__ float inv_freq_1d(float log_theta, int k, int dim) {
  // k in [0, dim/2)
  // inv_freq = 1 / theta^((2k)/dim)
  const float exponent = -log_theta * ((2.0f * (float)k) / (float)dim);
  return expf(exponent);
}

template <typename T>
__global__ void get_1d_rotary_pos_embed_kernel(
    T* __restrict__ out0, // cos OR real
    T* __restrict__ out1, // sin OR imag
    const T* __restrict__ pos_ptr, // optional; may be nullptr if pos is int
    const int pos_is_tensor, // 1 if pos_ptr valid, else 0
    const int64_t S, // sequence length
    const int dim, // input dim (must be even)
    const float theta,
    const int use_real, // 1/0
    const int repeat_interleave_real, // 1/0 (only meaningful if use_real=1)
    const float linear_factor,
    const float ntk_factor) {
  const int64_t row = (int64_t)blockIdx.x;
  if (row >= S) {
    return;
  }

  // pos: either from tensor or arange(S)
  float pos = 0.0f;
  if (pos_is_tensor) {
    pos = (float)LDG(&pos_ptr[row]);
  } else {
    pos = (float)row;
  }

  const float theta_eff = theta * ntk_factor;
  const float log_theta = logf(theta_eff);

  if (use_real) {
    // Output width is dim
    const int out_cols = dim;
    const int half = dim >> 1; // dim/2
    for (int c = (int)threadIdx.x; c < out_cols; c += (int)blockDim.x) {
      int k = 0;

      if (repeat_interleave_real) {
        // repeat_interleave: angles are keyed by k = c//2 (because columns are
        // duplicated)
        k = c >> 1; // in [0, dim/2)
      } else {
        // cat([cos, cos]) / cat([sin, sin]) where cos/sin originally [S, dim/2]
        // so we map c to k in [0, dim/2) by folding the second half
        k = (c < half) ? c : (c - half);
      }

      const float invf = inv_freq_1d(log_theta, k, dim) / linear_factor;
      const float ang = pos * invf;

      const float co = cosf(ang);
      const float si = sinf(ang);

      const int64_t base = row * (int64_t)out_cols;
      out0[base + (int64_t)c] = (T)co;
      out1[base + (int64_t)c] = (T)si;
    }
  } else {
    // Complex cis output: shape [S, dim/2] complex => return (real, imag) each
    // [S, dim/2]
    const int out_cols = dim >> 1;
    for (int c = (int)threadIdx.x; c < out_cols; c += (int)blockDim.x) {
      const int k = c; // in [0, dim/2)
      const float invf = inv_freq_1d(log_theta, k, dim) / linear_factor;
      const float ang = pos * invf;

      const float re = cosf(ang);
      const float im = sinf(ang);

      const int64_t base = row * (int64_t)out_cols;
      out0[base + (int64_t)c] = (T)re;
      out1[base + (int64_t)c] = (T)im;
    }
  }
}

template <typename T>
void invoke_get_1d_rotary_pos_embed(
    void* out0,
    void* out1,
    const void* pos_ptr, // may be nullptr
    int pos_is_tensor,
    int64_t S,
    int dim,
    float theta,
    int use_real,
    float linear_factor,
    float ntk_factor,
    int repeat_interleave_real,
    dinoml::DeviceStream stream) {
  const int threads = std::min(dim, 1024);
  dinoml::get_1d_rotary_pos_embed_kernel<T><<<S, threads, 0, stream>>>(
      static_cast<T*>(out0),
      static_cast<T*>(out1),
      static_cast<const T*>(pos_ptr),
      pos_is_tensor,
      S,
      dim,
      theta,
      use_real,
      repeat_interleave_real,
      linear_factor,
      ntk_factor);
}

} // namespace dinoml
