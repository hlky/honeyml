#pragma once

#include <cstdint>
#include <type_traits>

#include "dinoml/device.h"

namespace dinoml {

template <int Q, int... Ds>
struct is_in_dims : std::false_type {};
template <int Q, int D0, int... Ds>
struct is_in_dims<Q, D0, Ds...>
    : std::conditional_t<(Q == D0), std::true_type, is_in_dims<Q, Ds...>> {};

template <typename T, int Rank, int... FlipDims>
__device__ __forceinline__ int64_t
flip_index(int64_t out_linear, const int64_t (&sizes)[Rank]) {
  int64_t tmp = out_linear;
  int64_t coords[Rank];

#pragma unroll
  for (int i = Rank - 1; i >= 0; --i) {
    const int64_t s = sizes[i];
    const int64_t c = tmp % s;
    tmp /= s;
    if constexpr (is_in_dims<i, FlipDims...>::value) {
      coords[i] = (s - 1) - c;
    } else {
      coords[i] = c;
    }
  }

  int64_t in_linear = 0;
#pragma unroll
  for (int i = 0; i < Rank; ++i) {
    in_linear = in_linear * sizes[i] + coords[i];
  }
  return in_linear;
}

template <typename T, int Rank, int... FlipDims>
__global__ void flip_kernel(
    T* __restrict__ out,
    const T* __restrict__ in,
    const int64_t n,
    const int64_t* __restrict__ /*unused*/) {}

template <typename T, int Rank, int... FlipDims>
__global__ void flip_kernel_params(
    T* __restrict__ out,
    const T* __restrict__ in,
    const int64_t n,
    int64_t s0 = 1,
    int64_t s1 = 1,
    int64_t s2 = 1,
    int64_t s3 = 1,
    int64_t s4 = 1,
    int64_t s5 = 1,
    int64_t s6 = 1,
    int64_t s7 = 1,
    int64_t s8 = 1,
    int64_t s9 = 1,
    int64_t s10 = 1,
    int64_t s11 = 1,
    int64_t s12 = 1,
    int64_t s13 = 1,
    int64_t s14 = 1,
    int64_t s15 = 1) {
  int64_t sizes[Rank];

  if constexpr (Rank > 0)
    sizes[0] = s0;
  if constexpr (Rank > 1)
    sizes[1] = s1;
  if constexpr (Rank > 2)
    sizes[2] = s2;
  if constexpr (Rank > 3)
    sizes[3] = s3;
  if constexpr (Rank > 4)
    sizes[4] = s4;
  if constexpr (Rank > 5)
    sizes[5] = s5;
  if constexpr (Rank > 6)
    sizes[6] = s6;
  if constexpr (Rank > 7)
    sizes[7] = s7;
  if constexpr (Rank > 8)
    sizes[8] = s8;
  if constexpr (Rank > 9)
    sizes[9] = s9;
  if constexpr (Rank > 10)
    sizes[10] = s10;
  if constexpr (Rank > 11)
    sizes[11] = s11;
  if constexpr (Rank > 12)
    sizes[12] = s12;
  if constexpr (Rank > 13)
    sizes[13] = s13;
  if constexpr (Rank > 14)
    sizes[14] = s14;
  if constexpr (Rank > 15)
    sizes[15] = s15;

  for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += (int64_t)blockDim.x * gridDim.x) {
    const int64_t in_idx = flip_index<T, Rank, FlipDims...>(idx, sizes);
    out[idx] = LDG(&in[in_idx]);
  }
}

} // namespace dinoml

template <typename T, int Rank, int... FlipDims>
inline void invoke_flip(
    void* out,
    const void* in,
    int64_t n,
    const int64_t* sizes_host,
    dinoml::DeviceStream stream) {
  const int threads = 256;
  int64_t blocks = (n + threads - 1) / threads;
  if (blocks > 65535)
    blocks = 65535;

  int64_t s[16] = {1};
  for (int i = 0; i < Rank && i < 16; ++i)
    s[i] = sizes_host[i];

  dinoml::flip_kernel_params<T, Rank, FlipDims...><<<(int)blocks, threads, 0, stream>>>(
      static_cast<T*>(out),
      static_cast<const T*>(in),
      n,
      s[0],
      s[1],
      s[2],
      s[3],
      s[4],
      s[5],
      s[6],
      s[7],
      s[8],
      s[9],
      s[10],
      s[11],
      s[12],
      s[13],
      s[14],
      s[15]);
}
