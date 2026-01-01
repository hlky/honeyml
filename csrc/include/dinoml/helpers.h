#pragma once
#include <chrono>
#include <random>
#include "device.h"

namespace dinoml {

namespace helpers {

template <typename To, typename From>
struct convert;

template <>
struct convert<float2, float2> {
  __device__ __forceinline__ static float2 run(const float2& v) {
    return v;
  }
};

template <>
struct convert<float2, half2> {
  __device__ __forceinline__ static float2 run(const half2& v) {
    return __half22float2(v);
  }
};

template <>
struct convert<float2, bfloat162> {
  __device__ __forceinline__ static float2 run(const bfloat162& v) {
    return __bfloat1622float2(v);
  }
};

template <>
struct convert<half2, float2> {
  __device__ __forceinline__ static half2 run(const float2& v) {
    return __float22half2_rn(v);
  }
};

template <>
struct convert<half2, half2> {
  __device__ __forceinline__ static half2 run(const half2& v) {
    return v;
  }
};

template <>
struct convert<half2, bfloat162> {
  __device__ __forceinline__ static half2 run(const bfloat162& v) {
    return __float22half2_rn(__bfloat1622float2(v));
  }
};

template <>
struct convert<bfloat162, float2> {
  __device__ __forceinline__ static bfloat162 run(const float2& v) {
    return __float22bfloat162_rn(v);
  }
};

template <>
struct convert<bfloat162, half2> {
  __device__ __forceinline__ static bfloat162 run(const half2& v) {
    return __float22bfloat162_rn(__half22float2(v));
  }
};

template <>
struct convert<bfloat162, bfloat162> {
  __device__ __forceinline__ static bfloat162 run(const bfloat162& v) {
    return v;
  }
};

template <typename T>
struct add_op;

template <>
struct add_op<float2> {
  __device__ __forceinline__ static float2 run(float2 a, float2 b) {
    return {a.x + b.x, a.y + b.y};
  }
};

template <>
struct add_op<half2> {
  __device__ __forceinline__ static half2 run(half2 a, half2 b) {
    return __hadd2(a, b);
  }
};

template <>
struct add_op<bfloat162> {
  __device__ __forceinline__ static bfloat162 run(bfloat162 a, bfloat162 b) {
    return __hadd2(a, b);
  }
};

template <>
struct add_op<float> {
  __device__ __forceinline__ static float run(float a, float b) {
    return a + b;
  }
};

template <>
struct add_op<half> {
  __device__ __forceinline__ static half run(half a, half b) {
    return __hadd(a, b);
  }
};

template <>
struct add_op<bfloat16> {
  __device__ __forceinline__ static bfloat16 run(bfloat16 a, bfloat16 b) {
    return __hadd(a, b);
  }
};

template <typename T1, typename T2>
__device__ __forceinline__ T1 add2(T1 a, T2 b) {
  return add_op<T1>::run(a, convert<T1, T2>::run(b));
}

template <typename Vec, typename Elem>
struct add_op2; // Vec is storage type, Elem is logical element type

template <typename T>
struct add_op2<T, T> {
  __device__ __forceinline__ static T run(const T& a, const T& b) {
    return a + b;
  }
};

template <>
struct add_op2<float2, float> {
  __device__ __forceinline__ static float2 run(
      const float2& a,
      const float2& b) {
    return {a.x + b.x, a.y + b.y};
  }
};

template <>
struct add_op2<half2, half> {
  __device__ __forceinline__ static half2 run(const half2& a, const half2& b) {
    return __hadd2(a, b);
  }
};

template <>
struct add_op2<bfloat162, bfloat16> {
  __device__ __forceinline__ static bfloat162 run(
      const bfloat162& a,
      const bfloat162& b) {
    return __hadd2(a, b);
  }
};

template <typename Vec, typename Elem>
struct add_op2 {
  __device__ __forceinline__ static Vec run(const Vec& a, const Vec& b) {
    static_assert(
        sizeof(Vec) % sizeof(Elem) == 0,
        "Packed add requires Vec to be a multiple of Elem size.");
    Vec out;
    auto* po = reinterpret_cast<Elem*>(&out);
    auto* pa = reinterpret_cast<const Elem*>(&a);
    auto* pb = reinterpret_cast<const Elem*>(&b);

    constexpr int N = sizeof(Vec) / sizeof(Elem);

#pragma unroll
    for (int i = 0; i < N; ++i) {
      po[i] = pa[i] + pb[i];
    }
    return out;
  }
};

#ifdef DINOML_CUDA

constexpr uint32_t kBlockSizeBound = 256;
constexpr uint32_t kGridSizeBound = 4;

constexpr uint32_t kMaxGeneratorOffsetsPerCurandCall = 4;

static inline std::tuple<uint64_t, dim3, dim3> calc_execution_policy(
    int64_t total_elements,
    uint32_t unroll_factor) {
  const uint64_t numel = static_cast<uint64_t>(total_elements);

  const uint32_t block_size = kBlockSizeBound;
  dim3 block(block_size);

  uint32_t grid_x =
      static_cast<uint32_t>((numel + block_size - 1) / block_size);

  int dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, dev);

  uint32_t blocks_per_sm =
      static_cast<uint32_t>(prop.maxThreadsPerMultiProcessor / block_size);

  grid_x = std::min(prop.multiProcessorCount * blocks_per_sm, grid_x);

  dim3 grid(grid_x);

  const uint64_t denom = static_cast<uint64_t>(block_size) *
      static_cast<uint64_t>(grid_x) * static_cast<uint64_t>(unroll_factor);

  uint64_t counter_offset =
      ((numel - 1) / denom + 1) * kMaxGeneratorOffsetsPerCurandCall;

  return {counter_offset, grid, block};
}
#endif

template <typename integer>
constexpr __host__ __device__ inline integer ceil_div(integer n, integer m) {
  return (n + m - 1) / m;
}

inline uint64_t make_seed() {
  std::random_device rd;

  uint64_t time_seed = static_cast<uint64_t>(
      std::chrono::high_resolution_clock::now().time_since_epoch().count());

  uint64_t rd_seed = (static_cast<uint64_t>(rd()) << 32) ^ rd();

  return rd_seed ^ time_seed;
}

} // namespace helpers

} // namespace dinoml