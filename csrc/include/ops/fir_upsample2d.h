#pragma once
#include <dinoml/device.h>

namespace dinoml {

// Fixed FIR kernel: [1,3,3,1] normalized and scaled
__device__ __forceinline__ float fir_coeff(int i, int j) {
  const int a = (i == 0 || i == 3) ? 1 : 3;
  const int b = (j == 0 || j == 3) ? 1 : 3;
  return (float)(a * b) * (1.0f / 16.0f);
}

template <typename T>
__global__ void fir_upsample2d_kernel(
    T* __restrict__ out,
    const T* __restrict__ in,
    int N,
    int H,
    int W,
    int C) {
  const int OH = H * 2;
  const int OW = W * 2;

  const int idx = blockIdx.x;
  const int n = idx / (OH * OW);
  const int rem = idx - n * (OH * OW);
  const int oh = rem / OW;
  const int ow = rem % OW;

  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    float acc = 0.f;

// FIR convolution
#pragma unroll
    for (int fh = 0; fh < 4; ++fh) {
      int ih = oh - fh + 2;
      if ((unsigned)ih >= (unsigned)(2 * H))
        continue;
      if (ih & 1)
        continue; // skip non-sampled positions
      ih >>= 1;

#pragma unroll
      for (int fw = 0; fw < 4; ++fw) {
        int iw = ow - fw + 2;
        if ((unsigned)iw >= (unsigned)(2 * W))
          continue;
        if (iw & 1)
          continue;
        iw >>= 1;

        float w = fir_coeff(fh, fw);
        int idx_in = ((n * H + ih) * W + iw) * C + c;
        acc += w * (float)in[idx_in];
      }
    }

    int out_idx = ((n * OH + oh) * OW + ow) * C + c;
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
    dinoml::DeviceStream stream) {
  const int OH = H * 2;
  const int OW = W * 2;
  const int64_t blocks = (int64_t)N * OH * OW;
  const int threads = 256;

  dinoml::fir_upsample2d_kernel<T>
      <<<blocks, threads, 0, stream>>>((T*)out, (const T*)in, N, H, W, C);
}