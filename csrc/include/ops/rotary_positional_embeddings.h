#pragma once
#include <math.h>
#include <algorithm>
#include <cstdint>

#include "dinoml/device.h"

namespace dinoml {

// grid_type enum (kept in sync with python/operator):
// 0 = linspace
// 1 = slice
enum GridType3DRoPE : int {
  GRID_LINSPACE = 0,
  GRID_SLICE = 1,
};

__device__ __forceinline__ float rope_inv_freq(
    float log_theta,
    int k,
    int dim_axis) {
  // k is the rotary pair index (corresponding to positions [2k, 2k+1])
  // freq = 1 / theta^( (2k)/dim )
  const float exponent = -log_theta * ((2.0f * (float)k) / (float)dim_axis);
  return expf(exponent);
}

__device__ __forceinline__ float linspace_pos(
    float start,
    float stop,
    int size,
    int idx) {
  // torch.linspace(start, stop, steps=size)
  if (size <= 1) {
    return start;
  }
  const float t = (float)idx / (float)(size - 1);
  return start + (stop - start) * t;
}

template <typename T>
__global__ void get_3d_rotary_pos_embed_kernel(
    T* __restrict__ out_cos, // [N, embed_dim]
    T* __restrict__ out_sin, // [N, embed_dim]
    const int embed_dim,
    // crops_coords: (start_h, start_w), (stop_h, stop_w)
    const float crop_start_h,
    const float crop_start_w,
    const float crop_stop_h,
    const float crop_stop_w,
    // grid_size: (grid_size_h, grid_size_w)
    const int grid_size_h,
    const int grid_size_w,
    const int temporal_size,
    const float theta,
    const int grid_type, // 0=linspace, 1=slice
    const int max_h, // used only for slice
    const int max_w) { // used only for slice
  const int64_t idx = (int64_t)blockIdx.x; // over [0, N)
  const int64_t hw = (int64_t)grid_size_h * (int64_t)grid_size_w;
  const int64_t N = (int64_t)temporal_size * hw;
  if (idx >= N) {
    return;
  }

  const int t = (int)(idx / hw);
  const int hw_idx = (int)(idx - (int64_t)t * hw);

  const int h = hw_idx / grid_size_w;
  const int w = hw_idx - h * grid_size_w;

  // Positions
  float pos_t = (float)t;
  float pos_h = 0.0f;
  float pos_w = 0.0f;

  if (grid_type == (int)GRID_LINSPACE) {
    // match reference:
    // grid_h = linspace(start[0], stop[0] * (H-1)/H, H)
    // grid_w = linspace(start[1], stop[1] * (W-1)/W, W)
    const float stop_h =
        crop_stop_h * ((float)(grid_size_h - 1) / (float)grid_size_h);
    const float stop_w =
        crop_stop_w * ((float)(grid_size_w - 1) / (float)grid_size_w);
    pos_h = linspace_pos(crop_start_h, stop_h, grid_size_h, h);
    pos_w = linspace_pos(crop_start_w, stop_w, grid_size_w, w);
    // grid_t effectively linspace(0, temporal_size-1, temporal_size) == arange
    pos_t = (float)t;
  } else {
    // slice: positions are arange(max_h/max_w), then sliced by grid_size.
    // For our output, we only need the first grid_size_h/grid_size_w.
    // So pos_h = h, pos_w = w (assuming grid_size_* <= max_*)
    (void)max_h;
    (void)max_w;
    pos_h = (float)h;
    pos_w = (float)w;
    pos_t = (float)t;
  }

  // Axis dims (must match reference)
  const int dim_t = embed_dim / 4;
  const int dim_h = (embed_dim / 8) * 3;
  const int dim_w = (embed_dim / 8) * 3;

  const float log_theta = logf(theta);

  for (int d = (int)threadIdx.x; d < embed_dim; d += (int)blockDim.x) {
    float angle = 0.0f;

    if (d < dim_t) {
      const int k = d >> 1;
      const float invf = rope_inv_freq(log_theta, k, dim_t);
      angle = pos_t * invf;
    } else if (d < dim_t + dim_h) {
      const int dh = d - dim_t;
      const int k = dh >> 1;
      const float invf = rope_inv_freq(log_theta, k, dim_h);
      angle = pos_h * invf;
    } else {
      const int dw = d - (dim_t + dim_h);
      const int k = dw >> 1;
      const float invf = rope_inv_freq(log_theta, k, dim_w);
      angle = pos_w * invf;
    }

    const float c = cosf(angle);
    const float s = sinf(angle);

    const int64_t base = idx * (int64_t)embed_dim;
    out_cos[base + (int64_t)d] = (T)c;
    out_sin[base + (int64_t)d] = (T)s;
  }
}

template <typename T>
void invoke_get_3d_rotary_pos_embed(
    void* out_cos,
    void* out_sin,
    int embed_dim,
    float crop_start_h,
    float crop_start_w,
    float crop_stop_h,
    float crop_stop_w,
    int grid_size_h,
    int grid_size_w,
    int temporal_size,
    float theta,
    int grid_type,
    int max_h,
    int max_w,
    dinoml::DeviceStream stream) {
  const int64_t hw = (int64_t)grid_size_h * (int64_t)grid_size_w;
  const int64_t blocks = (int64_t)temporal_size * hw;
  const int threads = std::min(embed_dim, 1024);

  dinoml::get_3d_rotary_pos_embed_kernel<T><<<blocks, threads, 0, stream>>>(
      static_cast<T*>(out_cos),
      static_cast<T*>(out_sin),
      embed_dim,
      crop_start_h,
      crop_start_w,
      crop_stop_h,
      crop_stop_w,
      grid_size_h,
      grid_size_w,
      temporal_size,
      theta,
      grid_type,
      max_h,
      max_w);
}

template <typename T>
__global__ void get_2d_rotary_pos_embed_kernel(
    T* __restrict__ out_cos, // [HW, embed_dim]
    T* __restrict__ out_sin, // [HW, embed_dim]
    const int embed_dim,
    // crops_coords: (start_h,start_w), (stop_h,stop_w)
    const float crop_start_h,
    const float crop_start_w,
    const float crop_stop_h,
    const float crop_stop_w,
    // grid_size: (H, W) (note: python passes grid_size[0]=H, grid_size[1]=W)
    const int grid_h,
    const int grid_w,
    const float theta) {
  const int64_t idx = (int64_t)blockIdx.x; // [0, HW)
  const int64_t HW = (int64_t)grid_h * (int64_t)grid_w;
  if (idx >= HW) {
    return;
  }

  const int h = (int)(idx / (int64_t)grid_w);
  const int w = (int)(idx - (int64_t)h * (int64_t)grid_w);

  // positions:
  // grid_h uses start[0]..stop[0]*(H-1)/H ; grid_w uses
  // start[1]..stop[1]*(W-1)/W
  const float stop_h = crop_stop_h * ((float)(grid_h - 1) / (float)grid_h);
  const float stop_w = crop_stop_w * ((float)(grid_w - 1) / (float)grid_w);
  const float pos_h = linspace_pos(crop_start_h, stop_h, grid_h, h);
  const float pos_w = linspace_pos(crop_start_w, stop_w, grid_w, w);

  // each axis uses dim_axis = embed_dim/2
  const int dim_axis = embed_dim >> 1;
  const float log_theta = logf(theta);

  // output layout: [emb_h_axis, emb_w_axis] along dim
  for (int d = (int)threadIdx.x; d < embed_dim; d += (int)blockDim.x) {
    float angle = 0.0f;

    if (d < dim_axis) {
      // emb_h from grid[0] which is pos_w (meshgrid(w,h))
      const int k = d >> 1;
      const float invf = rope_inv_freq(log_theta, k, dim_axis);
      angle = pos_w * invf;
    } else {
      // emb_w from grid[1] which is pos_h
      const int da = d - dim_axis;
      const int k = da >> 1;
      const float invf = rope_inv_freq(log_theta, k, dim_axis);
      angle = pos_h * invf;
    }

    const float c = cosf(angle);
    const float s = sinf(angle);

    const int64_t base = idx * (int64_t)embed_dim;
    out_cos[base + (int64_t)d] = (T)c;
    out_sin[base + (int64_t)d] = (T)s;
  }
}

template <typename T>
void invoke_get_2d_rotary_pos_embed(
    void* out_cos,
    void* out_sin,
    int embed_dim,
    float crop_start_h,
    float crop_start_w,
    float crop_stop_h,
    float crop_stop_w,
    int grid_h,
    int grid_w,
    float theta,
    dinoml::DeviceStream stream) {
  const int64_t blocks = (int64_t)grid_h * (int64_t)grid_w;
  const int threads = std::min(embed_dim, 1024);

  dinoml::get_2d_rotary_pos_embed_kernel<T><<<blocks, threads, 0, stream>>>(
      static_cast<T*>(out_cos),
      static_cast<T*>(out_sin),
      embed_dim,
      crop_start_h,
      crop_start_w,
      crop_stop_h,
      crop_stop_w,
      grid_h,
      grid_w,
      theta);
}

__device__ __forceinline__ float rope_inv_freq_1d(
    float log_theta,
    int k,
    int dim) {
  const float exponent = -log_theta * ((2.0f * (float)k) / (float)dim);
  return expf(exponent);
}

template <typename T>
__global__ void get_2d_rotary_pos_embed_lumina_kernel(
    T* __restrict__ out_real, // [H, W, embed_dim/2]
    T* __restrict__ out_imag, // [H, W, embed_dim/2]
    const int embed_dim,
    const int len_h,
    const int len_w,
    const float linear_factor,
    const float ntk_factor,
    const float theta) {
  const int64_t idx = (int64_t)blockIdx.x; // over H*W
  const int64_t HW = (int64_t)len_h * (int64_t)len_w;
  if (idx >= HW) {
    return;
  }

  const int h = (int)(idx / (int64_t)len_w);
  const int w = (int)(idx - (int64_t)h * (int64_t)len_w);

  // 1D dim used in get_1d_rotary_pos_embed calls:
  // dim_1d = embed_dim/2
  const int dim_1d = embed_dim >> 1;
  const int out_cols = dim_1d; // final complex last-dim count (embed_dim/2)

  const float theta_eff = theta * ntk_factor;
  const float log_theta = logf(theta_eff);

  // output index base for [H, W, out_cols] contiguous
  const int64_t base =
      ((int64_t)h * (int64_t)len_w + (int64_t)w) * (int64_t)out_cols;

  // d in [0, out_cols) corresponds to flattened last two dims:
  // k in [0, dim_1d/2) and axis in {0,1}
  // d = k*2 + axis
  for (int d = (int)threadIdx.x; d < out_cols; d += (int)blockDim.x) {
    const int axis = d & 1; // 0 -> emb_h, 1 -> emb_w
    const int k = d >> 1; // frequency index in [0, dim_1d/2)
    const float pos = (axis == 0) ? (float)h : (float)w;

    // inv_freq per Lumina 1D complex path:
    // inv_freq = 1 / theta_eff^((2k)/dim_1d) / linear_factor
    const float invf = rope_inv_freq_1d(log_theta, k, dim_1d) / linear_factor;
    const float angle = pos * invf;

    const float re = cosf(angle);
    const float im = sinf(angle);

    out_real[base + (int64_t)d] = (T)re;
    out_imag[base + (int64_t)d] = (T)im;
  }
}

template <typename T>
void invoke_get_2d_rotary_pos_embed_lumina(
    void* out_real,
    void* out_imag,
    int embed_dim,
    int len_h,
    int len_w,
    float linear_factor,
    float ntk_factor,
    dinoml::DeviceStream stream) {
  const int64_t blocks = (int64_t)len_h * (int64_t)len_w;
  const int threads = std::min(std::max(embed_dim / 2, 1), 1024);
  const float theta = 10000.0f; // matches reference default

  dinoml::get_2d_rotary_pos_embed_lumina_kernel<T>
      <<<blocks, threads, 0, stream>>>(
          static_cast<T*>(out_real),
          static_cast<T*>(out_imag),
          embed_dim,
          len_h,
          len_w,
          linear_factor,
          ntk_factor,
          theta);
}

} // namespace dinoml
