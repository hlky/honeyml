#pragma once
#include <math.h>
#include <algorithm>
#include <cstdint>

#include "dinoml/device.h"

namespace dinoml {

// inv_freq(k) = 1 / theta^((2k)/dim_axis)
__device__ __forceinline__ float inv_freq_1d(
    float log_theta,
    int k,
    int dim_axis) {
  const float exponent = -log_theta * ((2.0f * (float)k) / (float)dim_axis);
  return expf(exponent);
}

// stable-audio / allegro 1D RoPE real outputs:
// freqs = outer(pos, inv_freq) where inv_freq length = dim_axis/2
// cos = cat([cos(freqs), cos(freqs)], dim=-1)  -> [S, dim_axis]
// sin = cat([sin(freqs), sin(freqs)], dim=-1)  -> [S, dim_axis]
template <typename T>
__global__ void rope_1d_allegro_kernel(
    T* __restrict__ out_cos, // [S, dim_axis]
    T* __restrict__ out_sin, // [S, dim_axis]
    const int S,
    const int dim_axis,
    const float theta,
    const float interpolation_scale) {
  const int s = (int)blockIdx.x;
  if (s >= S)
    return;

  const float pos = ((float)s) / interpolation_scale;

  const float log_theta = logf(theta);
  const int half = dim_axis >> 1; // dim_axis/2

  for (int d = (int)threadIdx.x; d < dim_axis; d += (int)blockDim.x) {
    // cat([x, x]) mapping
    const int k = (d < half) ? d : (d - half); // k in [0, half)
    const float invf = inv_freq_1d(log_theta, k, dim_axis);
    const float ang = pos * invf;

    out_cos[(int64_t)s * (int64_t)dim_axis + (int64_t)d] = (T)cosf(ang);
    out_sin[(int64_t)s * (int64_t)dim_axis + (int64_t)d] = (T)sinf(ang);
  }
}

// Build cartesian_prod(grid_t, grid_h, grid_w) where each grid is [0..len-1]
// long. Output layout matches: t slowest, w fastest: idx = t*(H*W) + h*W + w
__global__ void cartesian_prod_thw_kernel(
    int64_t* __restrict__ out_t, // [N]
    int64_t* __restrict__ out_h, // [N]
    int64_t* __restrict__ out_w, // [N]
    const int T,
    const int H,
    const int W) {
  const int64_t idx =
      (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  const int64_t HW = (int64_t)H * (int64_t)W;
  const int64_t N = (int64_t)T * HW;
  if (idx >= N)
    return;

  const int t = (int)(idx / HW);
  const int64_t rem = idx - (int64_t)t * HW;
  const int h = (int)(rem / (int64_t)W);
  const int w = (int)(rem - (int64_t)h * (int64_t)W);

  out_t[idx] = (int64_t)t;
  out_h[idx] = (int64_t)h;
  out_w[idx] = (int64_t)w;
}

template <typename T>
void invoke_get_3d_rotary_pos_embed_allegro(
    // freqs_t
    void* freqs_t_cos,
    void* freqs_t_sin,
    // freqs_h
    void* freqs_h_cos,
    void* freqs_h_sin,
    // freqs_w
    void* freqs_w_cos,
    void* freqs_w_sin,
    // grids (int64)
    void* grid_t,
    void* grid_h,
    void* grid_w,
    // params
    int height,
    int width,
    int num_frames,
    int vae_scale_factor_spatial,
    int patch_size,
    float interpolation_scale_h,
    float interpolation_scale_t,
    float interpolation_scale_w,
    int attention_head_dim,
    dinoml::DeviceStream stream) {
  // grid sizes
  const int grid_h_size = height / (vae_scale_factor_spatial * patch_size);
  const int grid_w_size = width / (vae_scale_factor_spatial * patch_size);
  const int F = num_frames;
  const int H = grid_h_size;
  const int W = grid_w_size;

  // dims per axis
  const int dim_t = attention_head_dim / 3;
  const int dim_h = attention_head_dim / 3;
  const int dim_w = attention_head_dim / 3;

  const float theta = 10000.0f;

  // 1D freqs kernels
  {
    const int threads_t = std::min(dim_t, 1024);
    dinoml::rope_1d_allegro_kernel<T><<<F, threads_t, 0, stream>>>(
        static_cast<T*>(freqs_t_cos),
        static_cast<T*>(freqs_t_sin),
        F,
        dim_t,
        theta,
        interpolation_scale_t);
  }
  {
    const int threads_h = std::min(dim_h, 1024);
    dinoml::rope_1d_allegro_kernel<T><<<H, threads_h, 0, stream>>>(
        static_cast<T*>(freqs_h_cos),
        static_cast<T*>(freqs_h_sin),
        H,
        dim_h,
        theta,
        interpolation_scale_h);
  }
  {
    const int threads_w = std::min(dim_w, 1024);
    dinoml::rope_1d_allegro_kernel<T><<<W, threads_w, 0, stream>>>(
        static_cast<T*>(freqs_w_cos),
        static_cast<T*>(freqs_w_sin),
        W,
        dim_w,
        theta,
        interpolation_scale_w);
  }

  // cartesian prod grids
  {
    const int64_t N = (int64_t)F * (int64_t)H * (int64_t)W;
    const int threads = 256;
    const int blocks = (int)((N + threads - 1) / threads);

    dinoml::cartesian_prod_thw_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<int64_t*>(grid_t),
        static_cast<int64_t*>(grid_h),
        static_cast<int64_t*>(grid_w),
        F,
        H,
        W);
  }
}

} // namespace dinoml
