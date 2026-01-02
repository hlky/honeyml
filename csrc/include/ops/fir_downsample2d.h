#pragma once
#include <dinoml/device.h>
#include <algorithm>
#include <cstdint>

namespace dinoml {

// Fixed FIR kernel (1,3,3,1) outer-product, normalized by 64.
// k2d[r][c] = v[r]*v[c] / 64, where v = [1,3,3,1]
__device__ __forceinline__ float fir_k2d_4x4(int r, int c) {
  // v: 1,3,3,1
  const int vr = (r == 0 || r == 3) ? 1 : 3;
  const int vc = (c == 0 || c == 3) ? 1 : 3;
  return (float)(vr * vc) * (1.0f / 64.0f);
}

// -------------------------
// No-conv path:
// out[n, oh, ow, c] = sum_{kh,kw} in[n, ih, iw, c] * k[kh,kw]
// ih = oh*2 + kh - 1, iw = ow*2 + kw - 1  (pad=1)
// zero-padding outside
// OH = H/2, OW = W/2 (assumes even H,W)
// -------------------------
template <typename T>
__global__ void fir_downsample2d_kernel_nhwc(
    T* __restrict__ out,
    const T* __restrict__ in,
    int N,
    int H,
    int W,
    int C) {
  const int OH = H >> 1;
  const int OW = W >> 1;

  // One block per (n, oh, ow)
  const int64_t block = (int64_t)blockIdx.x;
  const int64_t n = block / (int64_t)(OH * OW);
  const int64_t rem = block - n * (int64_t)(OH * OW);
  const int64_t oh = rem / OW;
  const int64_t ow = rem - oh * OW;

  for (int c = (int)threadIdx.x; c < C; c += (int)blockDim.x) {
    float acc = 0.0f;

// FIR 4x4
#pragma unroll
    for (int kh = 0; kh < 4; ++kh) {
      const int ih = (int)(oh * 2 + kh - 1);
      const bool h_in = (unsigned)ih < (unsigned)H;

#pragma unroll
      for (int kw = 0; kw < 4; ++kw) {
        const int iw = (int)(ow * 2 + kw - 1);
        const bool w_in = (unsigned)iw < (unsigned)W;

        if (h_in && w_in) {
          const int64_t in_idx =
              (((int64_t)n * H + ih) * W + iw) * (int64_t)C + c;
          const float x = (float)LDG(&in[in_idx]);
          acc += x * fir_k2d_4x4(kh, kw);
        }
      }
    }

    const int64_t out_idx = (((int64_t)n * OH + oh) * OW + ow) * (int64_t)C + c;
    out[out_idx] = (T)acc;
  }
}

// FIR kernel fixed: (1,3,3,1) normalized -> [1/8, 3/8, 3/8, 1/8]
__device__ __forceinline__ float fir_k1(int i) {
  // i in [0,3]
  return (i == 0 || i == 3) ? 0.125f : 0.375f;
}

template <typename T>
__global__ void fir_filter_pad2_kernel_nhwc(
    T* __restrict__ out,
    const T* __restrict__ in,
    int N, int H, int W, int C) {

  const int UH = H + 1;
  const int UW = W + 1;

  const int64_t block = (int64_t)blockIdx.x;
  const int64_t n  = block / (int64_t)(UH * UW);
  const int64_t rem = block - n * (int64_t)(UH * UW);
  const int64_t uh = rem / UW;
  const int64_t uw = rem - uh * UW;

  for (int c = (int)threadIdx.x; c < C; c += (int)blockDim.x) {
    float acc = 0.0f;

    // FIR 4x4 with pad=2: in index = (uh + fh - 2, uw + fw - 2)
    #pragma unroll
    for (int fh = 0; fh < 4; ++fh) {
      const int ih = (int)uh + fh - 2;
      if ((unsigned)ih >= (unsigned)H) continue;
      const float kh = fir_k1(fh);

      #pragma unroll
      for (int fw = 0; fw < 4; ++fw) {
        const int iw = (int)uw + fw - 2;
        if ((unsigned)iw >= (unsigned)W) continue;

        const float k = kh * fir_k1(fw);
        const int64_t in_idx = (((int64_t)n * H + ih) * W + iw) * (int64_t)C + c;
        acc += (float)LDG(&in[in_idx]) * k;
      }
    }

    const int64_t out_idx = (((int64_t)n * UH + uh) * UW + uw) * (int64_t)C + c;
    out[out_idx] = (T)acc;
  }
}

} // namespace dinoml

template <typename T>
void invoke_fir_filter_pad2(
    void* out,
    const void* in,
    int N, int H, int W, int C,
    dinoml::DeviceStream stream) {

  const int UH = H + 1;
  const int UW = W + 1;
  const int64_t blocks = (int64_t)N * UH * UW;
  const int threads = (int)std::min(C, 1024);

  dinoml::fir_filter_pad2_kernel_nhwc<T><<<blocks, threads, 0, stream>>>(
      (T*)out, (const T*)in, N, H, W, C);
}


template <typename T>
void invoke_fir_downsample2d(
    void* out,
    const void* in,
    int N,
    int H,
    int W,
    int C,
    dinoml::DeviceStream stream) {
  const int OH = H >> 1;
  const int OW = W >> 1;
  const int64_t blocks = (int64_t)N * OH * OW;
  const int threads = (int)std::min(C, 1024);

  dinoml::fir_downsample2d_kernel_nhwc<T>
      <<<blocks, threads, 0, stream>>>((T*)out, (const T*)in, N, H, W, C);
}
