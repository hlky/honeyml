#pragma once
#include <math.h>
#include <algorithm>
#include <cstdint>

#include "dinoml/device.h"

namespace dinoml {

template <typename T>
__device__ __forceinline__ float to_float(const T& x) {
  return (float)x;
}

template <typename TOut, typename TIn>
__global__ void get_timestep_embedding_kernel(
    TOut* __restrict__ out, // [N, embedding_dim]
    const TIn* __restrict__ timesteps, // [N]
    const int64_t n,
    const int embedding_dim,
    const bool flip_sin_to_cos,
    const float downscale_freq_shift,
    const float scale,
    const int max_period) {
  const int64_t t_idx = (int64_t)blockIdx.x;
  if (t_idx >= n) {
    return;
  }

  const int half_dim = embedding_dim / 2;

  // Handle degenerate case (embedding_dim == 1)
  if (half_dim <= 0) {
    if (threadIdx.x == 0) {
      out[t_idx * (int64_t)embedding_dim] = (TOut)0.0f;
    }
    return;
  }

  const float t = to_float(LDG(&timesteps[t_idx]));
  const float denom = (float)half_dim - downscale_freq_shift;
  const float inv_denom = 1.0f / denom;
  const float log_max_period = (float)log((float)max_period);

  for (int i = threadIdx.x; i < half_dim; i += blockDim.x) {
    // exponent = -log(max_period) * i / (half_dim - downscale_freq_shift)
    const float exponent = -log_max_period * (float)i * inv_denom;
    const float freq = expf(exponent);

    float arg = (t * freq) * scale;

    const float s = sinf(arg);
    const float c = cosf(arg);

    const int64_t base = t_idx * (int64_t)embedding_dim;
    if (!flip_sin_to_cos) {
      out[base + i] = (TOut)s;
      if (i + half_dim < embedding_dim) {
        out[base + i + half_dim] = (TOut)c;
      }
    } else {
      out[base + i] = (TOut)c;
      if (i + half_dim < embedding_dim) {
        out[base + i + half_dim] = (TOut)s;
      }
    }
  }

  // Zero-pad last column if embedding_dim is odd.
  if ((embedding_dim & 1) && threadIdx.x == 0) {
    out[t_idx * (int64_t)embedding_dim + (embedding_dim - 1)] = (TOut)0.0f;
  }
}

} // namespace dinoml

template <typename TOut, typename TIn>
void invoke_get_timestep_embedding(
    void* out,
    const void* timesteps,
    int64_t n,
    int embedding_dim,
    bool flip_sin_to_cos,
    float downscale_freq_shift,
    float scale,
    int max_period,
    dinoml::DeviceStream stream) {
  // One block per timestep.
  int64_t blocks = n;

  // Threads cover half_dim since we compute pairs (sin/cos) per i.
  int threads = std::min(std::max(embedding_dim / 2, 1), 1024);

  dinoml::get_timestep_embedding_kernel<TOut, TIn>
      <<<blocks, threads, 0, stream>>>(
          static_cast<TOut*>(out),
          static_cast<const TIn*>(timesteps),
          n,
          embedding_dim,
          flip_sin_to_cos,
          downscale_freq_shift,
          scale,
          max_period);
}
