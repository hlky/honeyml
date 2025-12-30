#pragma once
#include <math.h>
#include <algorithm>
#include <cstdint>

#include "dinoml/device.h"

namespace dinoml {

// compute 1 / 10000^(i / (half_dim))
__device__ __forceinline__ float inv_freq(int i, int half_dim) {
  // i in [0, half_dim)
  // exponent = i / half_dim
  const float exponent = (float)i / (float)half_dim;
  return powf(10000.0f, -exponent);
}

// d-part: returns sin/cos embedding for a single pos with embed_dim_1d (must be
// even). layout is [sin(0..half-1), cos(0..half-1)] where half = embed_dim_1d/2
__device__ __forceinline__ float sincos_1d(
    float pos,
    int embed_dim_1d,
    int d_in_1d) {
  const int half = embed_dim_1d >> 1;
  if (d_in_1d < half) {
    const float w = inv_freq(d_in_1d, half);
    return sinf(pos * w);
  } else {
    const int j = d_in_1d - half;
    const float w = inv_freq(j, half);
    return cosf(pos * w);
  }
}

template <typename T>
__global__ void sincos_pos_embed_3d_kernel(
    T* __restrict__ out,
    const int embed_dim,
    const int spatial_w, // spatial_size[0] in the reference code
    const int spatial_h, // spatial_size[1] in the reference code
    const int temporal_size,
    const float spatial_interpolation_scale,
    const float temporal_interpolation_scale) {
  const int64_t block_idx = (int64_t)blockIdx.x; // over (t, hw)
  const int64_t hw_size = (int64_t)spatial_w * (int64_t)spatial_h;

  const int t = (int)(block_idx / hw_size);
  const int hw = (int)(block_idx - (int64_t)t * hw_size);

  const int h = hw / spatial_w;
  const int w = hw - h * spatial_w;

  // positions (match the torch code: grid_w uses spatial_size[0], grid_h uses
  // spatial_size[1])
  const float pos_t = (float)t / temporal_interpolation_scale;
  const float pos_w =
      (float)w / spatial_interpolation_scale; // grid[0] in torch code
  const float pos_h =
      (float)h / spatial_interpolation_scale; // grid[1] in torch code

  // dims
  const int embed_dim_temporal = embed_dim / 4;
  const int embed_dim_spatial = 3 * embed_dim / 4;

  // spatial is 2D embed generated as:
  // emb = concat([emb_h(grid[0]=w), emb_w(grid[1]=h)]) with each 1D dim =
  // embed_dim_spatial/2
  const int embed_dim_1d_spatial = embed_dim_spatial / 2;

  for (int d = (int)threadIdx.x; d < embed_dim; d += (int)blockDim.x) {
    float val = 0.0f;

    if (d < embed_dim_temporal) {
      // temporal part: 1D with embed_dim_temporal
      val = sincos_1d(pos_t, embed_dim_temporal, d);
    } else {
      // spatial part: concat temporal first, then spatial
      const int ds = d - embed_dim_temporal; // in [0, embed_dim_spatial)
      if (ds < embed_dim_1d_spatial) {
        // first half from grid[0] (w) per reference implementation
        val = sincos_1d(pos_w, embed_dim_1d_spatial, ds);
      } else {
        // second half from grid[1] (h)
        val = sincos_1d(pos_h, embed_dim_1d_spatial, ds - embed_dim_1d_spatial);
      }
    }

    out[block_idx * (int64_t)embed_dim + (int64_t)d] = (T)val;
  }
}

template <typename T>
void invoke_sincos_pos_embed_3d(
    void* out,
    int embed_dim,
    int spatial_w,
    int spatial_h,
    int temporal_size,
    float spatial_interpolation_scale,
    float temporal_interpolation_scale,
    dinoml::DeviceStream stream) {
  const int64_t hw_size = (int64_t)spatial_w * (int64_t)spatial_h;
  const int64_t blocks = (int64_t)temporal_size * hw_size;
  const int threads = std::min(embed_dim, 1024);

  dinoml::sincos_pos_embed_3d_kernel<T><<<blocks, threads, 0, stream>>>(
      static_cast<T*>(out),
      embed_dim,
      spatial_w,
      spatial_h,
      temporal_size,
      spatial_interpolation_scale,
      temporal_interpolation_scale);
}

template <typename T>
__global__ void sincos_pos_embed_2d_kernel(
    T* __restrict__ out,
    const int embed_dim,
    const int grid_h, // grid_size[0] in reference code
    const int grid_w, // grid_size[1] in reference code
    const int cls_token, // 0/1
    const int extra_tokens,
    const float interpolation_scale,
    const int base_size) {
  const int64_t row = (int64_t)blockIdx.x; // each row in output
  const int prepend = (cls_token != 0 && extra_tokens > 0) ? extra_tokens : 0;

  // Optional prefix zeros
  if (row < prepend) {
    for (int d = (int)threadIdx.x; d < embed_dim; d += (int)blockDim.x) {
      out[row * (int64_t)embed_dim + (int64_t)d] = (T)0;
    }
    return;
  }

  const int64_t idx = row - (int64_t)prepend; // in [0, grid_w*grid_h)
  // flatten order matches torch meshgrid(grid_w, grid_h) flatten: w-major,
  // h-fastest
  const int w = (int)(idx % (int64_t)grid_w);
  const int h = (int)(idx / (int64_t)grid_w);

  // match reference scaling
  // grid_h: arange(H)/(H/base)/interp ; grid_w: arange(W)/(W/base)/interp
  const float scale_h =
      ((float)grid_h / (float)base_size) * interpolation_scale;
  const float scale_w =
      ((float)grid_w / (float)base_size) * interpolation_scale;

  const float pos_h = (float)h / scale_h; // grid[1]
  const float pos_w = (float)w / scale_w; // grid[0]

  // 2D uses embed_dim/2 for w-part then embed_dim/2 for h-part
  const int half_dim = embed_dim >> 1;

  for (int d = (int)threadIdx.x; d < embed_dim; d += (int)blockDim.x) {
    float val = 0.0f;
    if (d < half_dim) {
      // first half uses grid[0] (w) per reference
      // get_2d_sincos_pos_embed_from_grid
      val = sincos_1d(pos_w, half_dim, d);
    } else {
      // second half uses grid[1] (h)
      val = sincos_1d(pos_h, half_dim, d - half_dim);
    }
    out[row * (int64_t)embed_dim + (int64_t)d] = (T)val;
  }
}

template <typename T>
void invoke_sincos_pos_embed_2d(
    void* out,
    int embed_dim,
    int grid_h,
    int grid_w,
    int cls_token,
    int extra_tokens,
    float interpolation_scale,
    int base_size,
    dinoml::DeviceStream stream) {
  const int prepend = (cls_token != 0 && extra_tokens > 0) ? extra_tokens : 0;
  const int64_t rows = (int64_t)grid_h * (int64_t)grid_w + (int64_t)prepend;

  const int threads = std::min(embed_dim, 1024);
  dinoml::sincos_pos_embed_2d_kernel<T><<<rows, threads, 0, stream>>>(
      static_cast<T*>(out),
      embed_dim,
      grid_h,
      grid_w,
      cls_token,
      extra_tokens,
      interpolation_scale,
      base_size);
}

} // namespace dinoml
