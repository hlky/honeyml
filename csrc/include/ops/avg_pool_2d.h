#pragma once
#include <dinoml/device.h>
#include <dinoml/helpers.h>
#include <algorithm>
#include <cstdint>

namespace dinoml {

template <
    typename ElemType,
    typename VectorType,
    int KernelSize,
    int Stride,
    int Padding>
__global__ void avg_pool_2d_kernel(
    const ElemType* input_raw,
    ElemType* output_raw,
    const int N,
    const int H,
    const int W,
    const int C,
    const int HO,
    const int WO,
    const float norm_factor) {
  const VectorType* input = (const VectorType*)input_raw;
  VectorType* output = (VectorType*)output_raw;

  const int tid = threadIdx.x;
  const int n_idx = blockIdx.x;
  const int out_h_idx = blockIdx.y;
  const int out_w_idx = blockIdx.z;

  int h_start_idx = out_h_idx * Stride - Padding;
  int h_end_idx = h_start_idx + KernelSize;
  h_start_idx = (h_start_idx < 0) ? 0 : h_start_idx;
  h_end_idx = (h_end_idx > H) ? H : h_end_idx;

  int w_start_idx = out_w_idx * Stride - Padding;
  int w_end_idx = w_start_idx + KernelSize;
  w_start_idx = (w_start_idx < 0) ? 0 : w_start_idx;
  w_end_idx = (w_end_idx > W) ? W : w_end_idx;

  input += n_idx * H * W * C;
  output += ((n_idx * HO + out_h_idx) * WO + out_w_idx) * C;

  for (int c_idx = tid; c_idx < C; c_idx += blockDim.x) {
    float2 avg = {0.f, 0.f};

    for (int h = h_start_idx; h < h_end_idx; ++h) {
#pragma unroll
      for (int w = w_start_idx; w < w_end_idx; ++w) {
        const int idx = (h * W + w) * C;
        const VectorType tmp = LDG(input + (idx + c_idx));
        avg = dinoml::helpers::add2(avg, tmp);
      }
    }

    avg.x *= norm_factor;
    avg.y *= norm_factor;
    output[c_idx] = dinoml::helpers::convert<VectorType, float2>::run(avg);
  }
}

} // namespace dinoml

template <
    typename ElemType,
    typename VectorType,
    int KernelSize,
    int Stride,
    int Padding>
void avg_pool_2d_launcher(
    const ElemType* input,
    ElemType* output,
    const int N,
    const int H,
    const int W,
    const int C,
    const int HO,
    const int WO,
    dinoml::DeviceStream stream) {
  int num_thread = C / 2;
  if (num_thread > 256) {
    num_thread = 256;
  } else if (num_thread == 0) {
    num_thread = 1;
  }

  dim3 grid(N, HO, WO);
  dim3 block(num_thread);

  const float norm_factor =
      static_cast<float>(1.0f / (KernelSize * KernelSize));

  dinoml::avg_pool_2d_kernel<ElemType, VectorType, KernelSize, Stride, Padding>
      <<<grid, block, 0, stream>>>(input, output, N, H, W, C / 2, HO, WO, norm_factor);
}
