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

// -------------------------
// Conv-fused path:
// Implements (use_conv=True):
// 1) upfirdn2d with pad=(2,2), down=1 (i.e. FIR-filtered output of size (H+1,
// W+1)) 2) conv2d with kernel 3x3, stride=2, padding=0, plus bias
//
// Fused formula:
// out[n,oh,ow,oc] = bias[oc] + sum_ic sum_ky,kx W[oc, ic, ky, kx] * U[n, ic,
// ih, iw] where ih=oh*2+ky, iw=ow*2+kx, and U[n,ic,ih,iw] = sum_fh,fw in[n,ic,
// (ih+fh-2), (iw+fw-2)] * fir_k2d_4x4(fh,fw)
//
// We assume weight layout is **HWIO** for NHWC conv convenience:
// weight[ky, kx, ic, oc]
// bias[oc]
// -------------------------
template <typename T, typename Wt>
__global__ void fir_downsample2d_conv_kernel_nhwc_hwio(
    T* __restrict__ out,
    const T* __restrict__ in,
    const Wt* __restrict__ weight, // [3,3,C,OC]
    const T* __restrict__ bias, // [OC]
    int N,
    int H,
    int W,
    int C,
    int OC) {
  const int OH = H >> 1;
  const int OW = W >> 1;

  const int64_t block = (int64_t)blockIdx.x;
  const int64_t n = block / (int64_t)(OH * OW);
  const int64_t rem = block - n * (int64_t)(OH * OW);
  const int64_t oh = rem / OW;
  const int64_t ow = rem - oh * OW;

  for (int oc = (int)threadIdx.x; oc < OC; oc += (int)blockDim.x) {
    float acc = (bias != nullptr) ? (float)LDG(&bias[oc]) : 0.0f;

// conv 3x3, stride=2
#pragma unroll
    for (int ky = 0; ky < 3; ++ky) {
      const int uh = (int)(oh * 2 + ky); // index in U (size H+1)
// uh valid: [0..H]
// U computed from in with FIR pad=2 => in index (uh+fh-2)
#pragma unroll
      for (int kx = 0; kx < 3; ++kx) {
        const int uw = (int)(ow * 2 + kx); // index in U (size W+1)

        // For each input channel, compute U value and multiply with weight
        for (int ic = 0; ic < C; ++ic) {
          float u = 0.0f;

// FIR 4x4 with pad=2
#pragma unroll
          for (int fh = 0; fh < 4; ++fh) {
            const int ih = uh + fh - 2;
            const bool h_in = (unsigned)ih < (unsigned)H;

#pragma unroll
            for (int fw = 0; fw < 4; ++fw) {
              const int iw = uw + fw - 2;
              const bool w_in = (unsigned)iw < (unsigned)W;

              if (h_in && w_in) {
                const int64_t in_idx =
                    (((int64_t)n * H + ih) * W + iw) * (int64_t)C + ic;
                const float x = (float)LDG(&in[in_idx]);
                u += x * fir_k2d_4x4(fh, fw);
              }
            }
          }

          // weight HWIO: [ky,kx,ic,oc]
          const int64_t w_idx =
              (((int64_t)ky * 3 + kx) * (int64_t)C + ic) * (int64_t)OC + oc;
          const float wv = (float)LDG(&weight[w_idx]);

          acc += u * wv;
        }
      }
    }

    const int64_t out_idx =
        (((int64_t)n * OH + oh) * OW + ow) * (int64_t)OC + oc;
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

  dinoml::fir_downsample2d_conv_kernel_nhwc_hwio<T, Wt>
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
