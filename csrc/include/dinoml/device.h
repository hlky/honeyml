#pragma once

#include <chrono>
#include <random>
#ifdef DINOML_CUDA
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "cutlass/util/host_tensor.h"
#endif
#ifdef DINOML_HIP
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include "library/include/ck/library/utility/host_tensor.hpp"
#endif

namespace dinoml {
#ifdef DINOML_CUDA
using bfloat16 = __nv_bfloat16;
using DeviceStream = cudaStream_t;
#endif
#ifdef DINOML_HIP
using bfloat16 = hip_bfloat16;
using DeviceStream = hipStream_t;
#endif

inline uint64_t make_seed() {
  std::random_device rd;

  uint64_t time_seed = static_cast<uint64_t>(
      std::chrono::high_resolution_clock::now().time_since_epoch().count());

  uint64_t rd_seed = (static_cast<uint64_t>(rd()) << 32) ^ rd();

  return rd_seed ^ time_seed;
}

} // namespace dinoml

#ifdef DINOML_CUDA
using bfloat16 = __nv_bfloat16;
#endif
#ifdef DINOML_HIP
using bfloat16 = hip_bfloat16;
#endif

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
