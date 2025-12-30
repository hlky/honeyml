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

} // namespace dinoml
