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

template <typename T, typename Wt>
__global__ void fir_downsample2d_conv_kernel_nhwc_ohwi(
    T* __restrict__ out,
    const T* __restrict__ in,
    const Wt* __restrict__ weight, // [OC, KH, KW, C]
    const T* __restrict__ bias, // [OC]
    int N,
    int H,
    int W,
    int C,
    int OC) {
  const int OH = H >> 1;
  const int OW = W >> 1;

  const int block = blockIdx.x;
  const int n = block / (OH * OW);
  const int rem = block - n * (OH * OW);
  const int oh = rem / OW;
  const int ow = rem - oh * OW;

  // Precompute FIR kernel (constant)
  constexpr float k[4] = {1.f / 8.f, 3.f / 8.f, 3.f / 8.f, 1.f / 8.f};

  for (int oc = threadIdx.x; oc < OC; oc += blockDim.x) {
    float acc = bias ? (float)bias[oc] : 0.0f;

// ---- convolution ----
#pragma unroll
    for (int ky = 0; ky < 3; ++ky) {
      const int uh = oh * 2 + ky;

#pragma unroll
      for (int kx = 0; kx < 3; ++kx) {
        const int uw = ow * 2 + kx;

        // FIR + channel accumulation
        float sum = 0.f;

#pragma unroll
        for (int fh = 0; fh < 4; ++fh) {
          const int ih = uh + fh - 2;
          if ((unsigned)ih >= (unsigned)H)
            continue;

#pragma unroll
          for (int fw = 0; fw < 4; ++fw) {
            const int iw = uw + fw - 2;
            if ((unsigned)iw >= (unsigned)W)
              continue;

            const float kf = k[fh] * k[fw];
            const int base = ((n * H + ih) * W + iw) * C;

#pragma unroll
            for (int ic = 0; ic < C; ++ic) {
              sum += (float)in[base + ic] * kf *
                  (float)weight[((oc * 3 + ky) * 3 + kx) * C + ic];
            }
          }
        }

        acc += sum;
      }
    }

    const int out_idx = ((n * OH + oh) * OW + ow) * OC + oc;
    out[out_idx] = (T)acc;
  }
}

} // namespace dinoml

template <typename T, typename Wt>
void invoke_fir_downsample2d_conv(
    void* out,
    const void* in,
    const void* weight,
    const void* bias,
    int N,
    int H,
    int W,
    int C,
    int OC,
    dinoml::DeviceStream stream) {
  const int OH = H >> 1;
  const int OW = W >> 1;
  const int64_t blocks = (int64_t)N * OH * OW;
  const int threads = (int)std::min(OC, 1024);

  dinoml::fir_downsample2d_conv_kernel_nhwc_ohwi<T, Wt>
      <<<blocks, threads, 0, stream>>>(
          (T*)out,
          (const T*)in,
          (const Wt*)weight,
          (const T*)bias,
          N,
          H,
          W,
          C,
          OC);
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
