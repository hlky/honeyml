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
  const int OH = H * up;
  const int OW = W * up;

  const int64_t block = (int64_t)blockIdx.x;
  const int64_t n = block / (int64_t)(OH * OW);
  const int64_t rem = block - n * (int64_t)(OH * OW);
  const int64_t oh = rem / OW;
  const int64_t ow = rem % OW;

  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    float acc = 0.f;

#pragma unroll
    for (int fh = 0; fh < 4; ++fh) {
      int py = oh + fh - pad0;
      if ((unsigned)py >= (unsigned)(OH + pad1))
        continue;
      if (py & 1)
        continue;
      int iy = py >> 1;
      if ((unsigned)iy >= (unsigned)H)
        continue;

#pragma unroll
      for (int fw = 0; fw < 4; ++fw) {
        int px = ow + fw - pad0;
        if ((unsigned)px >= (unsigned)(OW + pad1))
          continue;
        if (px & 1)
          continue;
        int ix = px >> 1;
        if ((unsigned)ix >= (unsigned)W)
          continue;

        const int64_t idx = ((int64_t)n * H + iy) * W + ix;
        acc += (float)in[idx * C + c] * k2d(fh, fw);
      }
    }

    out[((n * OH + oh) * OW + ow) * C + c] = (T)acc;
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
  const int OH = H * up;
  const int OW = W * up;
  const int64_t blocks = (int64_t)N * OH * OW;
  const int threads = (int)std::min(C, 1024);

  dinoml::fir_upsample2d_kernel<T>
      <<<blocks, threads, 0, stream>>>((T*)out, (const T*)in, N, H, W, C, up, pad0, pad1);
}