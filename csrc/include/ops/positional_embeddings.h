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
__global__ void sincos_pos_embed_3d_cogvideox_joint_kernel(
    T* __restrict__ out, // [1, total_len, D] contiguous
    const int embed_dim,
    const int spatial_w,
    const int spatial_h,
    const int temporal_size,
    const int max_text_seq_length,
    const float spatial_interpolation_scale,
    const float temporal_interpolation_scale) {
  const int64_t out_pos = (int64_t)blockIdx.x; // in [0, total_len)
  const int64_t hw_size = (int64_t)spatial_w * (int64_t)spatial_h;
  const int64_t num_patches = (int64_t)temporal_size * hw_size;
  const int64_t total_len = (int64_t)max_text_seq_length + num_patches;

  if (out_pos >= total_len) {
    return;
  }

  // If this is within the text prefix, it's zero.
  if (out_pos < (int64_t)max_text_seq_length) {
    for (int d = (int)threadIdx.x; d < embed_dim; d += (int)blockDim.x) {
      out[out_pos * (int64_t)embed_dim + (int64_t)d] = (T)0;
    }
    return;
  }

  // Otherwise, map into patch index p in [0, num_patches)
  const int64_t p = out_pos - (int64_t)max_text_seq_length;
  const int t = (int)(p / hw_size);
  const int hw = (int)(p - (int64_t)t * hw_size);

  const int h = hw / spatial_w;
  const int w = hw - h * spatial_w;

  const float pos_t = (float)t / temporal_interpolation_scale;
  const float pos_w = (float)w / spatial_interpolation_scale; // grid[0] (w)
  const float pos_h = (float)h / spatial_interpolation_scale; // grid[1] (h)

  const int embed_dim_temporal = embed_dim / 4;
  const int embed_dim_spatial = 3 * embed_dim / 4;
  const int embed_dim_1d_spatial = embed_dim_spatial / 2;

  for (int d = (int)threadIdx.x; d < embed_dim; d += (int)blockDim.x) {
    float val = 0.0f;

    if (d < embed_dim_temporal) {
      val = sincos_1d(pos_t, embed_dim_temporal, d);
    } else {
      const int ds = d - embed_dim_temporal;
      if (ds < embed_dim_1d_spatial) {
        val = sincos_1d(pos_w, embed_dim_1d_spatial, ds);
      } else {
        val = sincos_1d(pos_h, embed_dim_1d_spatial, ds - embed_dim_1d_spatial);
      }
    }

    out[out_pos * (int64_t)embed_dim + (int64_t)d] = (T)val;
  }
}

template <typename T>
void invoke_sincos_pos_embed_3d_cogvideox_joint(
    void* out,
    int embed_dim,
    int spatial_w,
    int spatial_h,
    int temporal_size,
    int max_text_seq_length,
    float spatial_interpolation_scale,
    float temporal_interpolation_scale,
    dinoml::DeviceStream stream) {
  const int64_t hw_size = (int64_t)spatial_w * (int64_t)spatial_h;
  const int64_t num_patches = (int64_t)temporal_size * hw_size;
  const int64_t total_len = (int64_t)max_text_seq_length + num_patches;

  const int64_t blocks = total_len; // one block per output position
  const int threads = std::min(embed_dim, 1024);

  dinoml::sincos_pos_embed_3d_cogvideox_joint_kernel<T>
      <<<blocks, threads, 0, stream>>>(
          static_cast<T*>(out),
          embed_dim,
          spatial_w,
          spatial_h,
          temporal_size,
          max_text_seq_length,
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

  const int64_t idx = row - (int64_t)prepend;
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
      val = sincos_1d(pos_w, half_dim, d);
    } else {
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

template <typename T>
__global__ void cogview3plus_joint_pos_embed_kernel(
    T* __restrict__ out, // [1, text_length + H*W, D]
    const T* __restrict__ pos_table, // [P, P, D] contiguous
    const int hidden_size,
    const int pos_embed_max_size, // P
    const int height, // H
    const int width, // W
    const int text_length) {
  const int64_t row = (int64_t)blockIdx.x; // [0, text_length + H*W)
  const int64_t hw = (int64_t)height * (int64_t)width;
  const int64_t total = (int64_t)text_length + hw;

  if (row >= total) {
    return;
  }

  // Text prefix is zeros
  if (row < (int64_t)text_length) {
    for (int d = (int)threadIdx.x; d < hidden_size; d += (int)blockDim.x) {
      out[row * (int64_t)hidden_size + (int64_t)d] = (T)0;
    }
    return;
  }

  // Image positions: flatten row-major (h major then w), matching:
  //   image_pos_embed = pos_table[:H, :W].reshape(H*W, -1)
  const int64_t p = row - (int64_t)text_length; // in [0, H*W)
  const int h = (int)(p / (int64_t)width);
  const int w = (int)(p - (int64_t)h * (int64_t)width);

  // pos_table is indexed [h, w, d] with leading dim P (pos_embed_max_size)
  const int64_t base = ((int64_t)h * (int64_t)pos_embed_max_size + (int64_t)w) *
      (int64_t)hidden_size;

  for (int d = (int)threadIdx.x; d < hidden_size; d += (int)blockDim.x) {
    out[row * (int64_t)hidden_size + (int64_t)d] =
        LDG(&pos_table[base + (int64_t)d]);
  }
}

template <typename T>
void invoke_cogview3plus_joint_pos_embed(
    void* out,
    const void* pos_table,
    int hidden_size,
    int pos_embed_max_size,
    int height,
    int width,
    int text_length,
    dinoml::DeviceStream stream) {
  const int64_t hw = (int64_t)height * (int64_t)width;
  const int64_t rows = (int64_t)text_length + hw;

  const int threads = std::min(hidden_size, 1024);
  dinoml::cogview3plus_joint_pos_embed_kernel<T><<<rows, threads, 0, stream>>>(
      static_cast<T*>(out),
      static_cast<const T*>(pos_table),
      hidden_size,
      pos_embed_max_size,
      height,
      width,
      text_length);
}

template <typename T>
__global__ void cropped_pos_embed_kernel(
    T* __restrict__ out,
    const int embed_dim,
    const int pos_embed_max_size, // M
    const int base_size,
    const float interpolation_scale,
    const int patch_size,
    const int height,
    const int width) {
  const int Hc = height / patch_size;
  const int Wc = width / patch_size;

  const int top = (pos_embed_max_size - Hc) / 2;
  const int left = (pos_embed_max_size - Wc) / 2;

  const int64_t idx = (int64_t)blockIdx.x; // [0, Hc*Wc)
  const int w_local = (int)(idx % (int64_t)Wc);
  const int h_local = (int)(idx / (int64_t)Wc);

  const int h = top + h_local;
  const int w = left + w_local;

  // match get_2d_sincos_pos_embed scaling for grid_size=(M,M)
  const float scale =
      ((float)pos_embed_max_size / (float)base_size) * interpolation_scale;
  const float pos_h = (float)h / scale;
  const float pos_w = (float)w / scale;

  const int half_dim = embed_dim >> 1;

  for (int d = (int)threadIdx.x; d < embed_dim; d += (int)blockDim.x) {
    float val = 0.0f;
    if (d < half_dim) {
      // first half uses grid[0] == w
      val = sincos_1d(pos_w, half_dim, d);
    } else {
      // second half uses grid[1] == h
      val = sincos_1d(pos_h, half_dim, d - half_dim);
    }
    // out is shaped [1, Hc*Wc, D] -> contiguous like [Hc*Wc, D]
    out[idx * (int64_t)embed_dim + (int64_t)d] = (T)val;
  }
}

template <typename T>
void invoke_cropped_pos_embed(
    void* out,
    int embed_dim,
    int pos_embed_max_size,
    int base_size,
    float interpolation_scale,
    int patch_size,
    int height,
    int width,
    dinoml::DeviceStream stream) {
  const int Hc = height / patch_size;
  const int Wc = width / patch_size;
  const int64_t blocks = (int64_t)Hc * (int64_t)Wc;
  const int threads = std::min(embed_dim, 1024);

  dinoml::cropped_pos_embed_kernel<T><<<blocks, threads, 0, stream>>>(
      static_cast<T*>(out),
      embed_dim,
      pos_embed_max_size,
      base_size,
      interpolation_scale,
      patch_size,
      height,
      width);
}

} // namespace dinoml
