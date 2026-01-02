#pragma once
#include <dinoml/device.h>

namespace dinoml {
// v=[1,3,3,1]; outer sum=64; scaled by factor^2=4 => scale=4/64 = 1/16
__device__ __forceinline__ float k2d(int r, int c) {
  const int vr = (r == 0 || r == 3) ? 1 : 3;
  const int vc = (c == 0 || c == 3) ? 1 : 3;
  return (float)(vr * vc) * (1.0f / 16.0f);
}

template <typename T>
__global__ void fir_upsample2d_kernel(
    T* __restrict__ out,
    const T* __restrict__ in,
    int N,
    int H,
    int W,
    int C,
    int up,
    int pad0,
    int pad1) {
  constexpr int KH = 4;
  constexpr int KW = 4;

  // Output size matches upfirdn2d_native (down=1): H*up + pad0 + pad1 - K + 1
  const int OH = H * up + pad0 + pad1 - KH + 1;
  const int OW = W * up + pad0 + pad1 - KW + 1;

  const int64_t block = (int64_t)blockIdx.x;
  const int64_t n = block / (int64_t)(OH * OW);
  const int64_t rem = block - n * (int64_t)(OH * OW);
  const int64_t oh = rem / OW;
  const int64_t ow = rem - oh * OW;

  for (int c = (int)threadIdx.x; c < C; c += (int)blockDim.x) {
    float acc = 0.0f;

#pragma unroll
    for (int fh = 0; fh < KH; ++fh) {
      const int uy = (int)oh + fh - pad0; // coord in upsampled (unpadded) grid
      // Must land on an original sample location
      if (uy % up != 0)
        continue;
      const int iy = uy / up;
      if ((unsigned)iy >= (unsigned)H)
        continue;

#pragma unroll
      for (int fw = 0; fw < KW; ++fw) {
        const int ux = (int)ow + fw - pad0;
        if (ux % up != 0)
          continue;
        const int ix = ux / up;
        if ((unsigned)ix >= (unsigned)W)
          continue;

        const int64_t in_idx =
            (((int64_t)n * H + iy) * W + ix) * (int64_t)C + c;
        acc += (float)LDG(&in[in_idx]) * k2d(fh, fw);
      }
    }

    const int64_t out_idx = (((int64_t)n * OH + oh) * OW + ow) * (int64_t)C + c;
    out[out_idx] = (T)acc;
  }
}

} // namespace dinoml

template <typename T>
void invoke_fir_upsample2d(
    void* out,
    const void* in,
    int N,
    int H,
    int W,
    int C,
    int up,
    int pad0,
    int pad1,
    dinoml::DeviceStream stream) {
  constexpr int KH = 4;
  constexpr int KW = 4;

  const int OH = H * up + pad0 + pad1 - KH + 1;
  const int OW = W * up + pad0 + pad1 - KW + 1;

  const int64_t blocks = (int64_t)N * OH * OW;
  const int threads = (int)std::min(C, 1024);

  dinoml::fir_upsample2d_kernel<T><<<blocks, threads, 0, stream>>>(
      (T*)out, (const T*)in, N, H, W, C, up, pad0, pad1);
}