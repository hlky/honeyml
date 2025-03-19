//  Copyright 2025 hlky. All rights reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
#pragma once
#include <honey/device.h>
#include <stddef.h>
#include <stdint.h>
#include <array>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>
#include "short_file.h"

// We compile all models with -fvisibility=hidden. Any symbols that need to be
// exposed in the final shared library must be declared with Honey_EXPORT to make
// them visible.

#ifdef __GNUC__ // Applies to any compiler with GNU extensions (clang and g++)
#define Honey_EXPORT __attribute__((__visibility__("default")))
#else
#ifdef _WIN32
#define Honey_EXPORT __declspec(dllexport)
#else
#define Honey_EXPORT
#endif
#endif

struct HoneyModelOpaque {};
using HoneyModelHandle = HoneyModelOpaque*;

enum class HoneyWorkspaceAllocationMode {
  // workspace is allocated immediately and not released until module is
  // unloaded
  kEager = 0,
  // workspace is allocated at first run and not released until module is
  // unloaded
  kLazy,
  // workspace is allocated each run then freed after use
  kFau
};

enum class HoneyError : int {
  HoneySuccess = 0,
  HoneyFailure = 1,
};

#define Honey_ERROR_CHECK(call)                                        \
  if ((call) != HoneyError::HoneySuccess) {                \
    throw std::runtime_error(                                        \
        std::string(#call " API call failed at ") + __SHORT_FILE__ + \
        ", line" + std::to_string(__LINE__));                        \
  }

struct HoneyParamShape {
  HoneyParamShape() : shape_data(nullptr), size(0) {}
  HoneyParamShape(const int64_t* shape_data_in, size_t size_in)
      : shape_data(shape_data_in), size(size_in) {}

  const int64_t* shape_data;
  size_t size;

  size_t Numel() const {
    return std::accumulate(
        shape_data, shape_data + size, (int64_t)1, std::multiplies<int64_t>());
  }
};

enum class HoneyDtype {
  kUnset = 0,
  kHalf,
  kFloat,
  kInt,
  kLong,
  kBool,
  kBFloat16,
  kFloat8_e4m3,
  kFloat8_e5m2,
};

struct HoneyData {
  HoneyData() : ptr(nullptr), dtype(HoneyDtype::kUnset) {}

  HoneyData(
      void* ptr_in,
      const HoneyParamShape& shape_in,
      HoneyDtype dtype_in)
      : ptr(ptr_in), shape(shape_in), dtype(dtype_in) {}

  void* ptr;
  HoneyParamShape shape;
  HoneyDtype dtype;
};

inline size_t HoneyDtypeSizeBytes(HoneyDtype dtype) {
  switch (dtype) {
    case HoneyDtype::kHalf:
    case HoneyDtype::kBFloat16:
      return 2;
    case HoneyDtype::kFloat:
      return 4;
    case HoneyDtype::kInt:
      return 4;
    case HoneyDtype::kLong:
      return 8;
    case HoneyDtype::kFloat8_e4m3:
    case HoneyDtype::kFloat8_e5m2:
    case HoneyDtype::kBool:
      return 1;
    case HoneyDtype::kUnset:
      throw std::runtime_error("Unset dtype has no size!");
  }
  throw std::runtime_error("dtype handling is not implemented!");
}

struct HoneyStreamOpaque {};
using HoneyStreamHandle = HoneyStreamOpaque*;

// Allocator to use for GPU mallocs and frees. Allocations will only happen
// when the ModelContainer is created.
class HoneyAllocator {
 public:
  virtual void* Allocate(size_t nbytes) = 0;
  virtual void Free(void* ptr) = 0;

  virtual ~HoneyAllocator() = default;
};

// Some custom allocators are provided. They can be created by passing
// an enum into the HoneyAllocatorCreate() function.
enum class HoneyAllocatorType {
  // The default allocator just uses the backend's default malloc/free.
  kDefault = 0,
  // The tracking allocator is like the default allocator, but it keeps
  // track of how many bytes it has allocated. Mainly used for testing.
  kTracking,
};

extern "C" {

Honey_EXPORT const char* GetLastErrorMessage();

// Create a ModelContainer. See model_container.h for all the details.
// Some important high-level notes:
// * If allocator is null, a default allocator is used (forwards to
//   {cuda/hip}{Malloc/Free}).
// * We assume that the allocator lives at least as long as the ModelContainer.
Honey_EXPORT HoneyError HoneyModelContainerCreate(
    HoneyModelHandle* ret,
    size_t num_runtimes,
    HoneyAllocator* allocator = nullptr);

Honey_EXPORT HoneyError
HoneyModelContainerDelete(HoneyModelHandle handle);

Honey_EXPORT HoneyError HoneyStreamCreate(
  HoneyStreamHandle* stream_handle,
  bool non_blocking);

Honey_EXPORT HoneyError HoneyModelContainerSetConstant(
    HoneyModelHandle handle,
    const char* name,
    const HoneyData* tensor);

Honey_EXPORT HoneyError HoneyModelContainerSetManyConstants(
    HoneyModelHandle handle,
    const char** names,
    const HoneyData* tensors,
    size_t num_tensors);

Honey_EXPORT HoneyError HoneyModelContainerSetDoubleBufferConstant(
    HoneyModelHandle handle,
    HoneyStreamHandle stream_handle,
    const char* name,
    const HoneyData* tensor);

Honey_EXPORT HoneyError HoneyModelContainerSetManyDoubleBufferConstants(
    HoneyModelHandle handle,
    HoneyStreamHandle stream_handle,
    const char** names,
    const HoneyData* tensors,
    size_t num_tensors);

Honey_EXPORT HoneyError HoneyModelContainerGetNumConstants(
    HoneyModelHandle handle,
    bool unbound_constants_only,
    bool constant_folding_inputs_only,
    size_t* num_constants_out);

Honey_EXPORT HoneyError HoneyModelContainerGetConstantNames(
    HoneyModelHandle handle,
    bool unbound_constants_only,
    bool constant_folding_inputs_only,
    const char** constant_names_out);

Honey_EXPORT HoneyError HoneyModelContainerGetConstantDtype(
    HoneyModelHandle handle,
    const char* name,
    HoneyDtype* dtype);

Honey_EXPORT HoneyError HoneyModelContainerGetConstantOriginalName(
    HoneyModelHandle handle,
    const char* name,
    const char** original_name_out);

Honey_EXPORT HoneyError HoneyModelContainerRun(
    HoneyModelHandle handle,
    const HoneyData* inputs,
    size_t num_inputs,
    HoneyData* outputs,
    size_t num_outputs,
    HoneyStreamHandle stream_handle,
    bool sync,
    bool graph_mode,
    int64_t** output_shapes_out);

// Like HoneyModelContainerRun, but expects outputs to be allocated on the
// host. Does an extra sync/copy at the end to copy them over. Warning: don't
// use this! It's not optimal with respect to performance. It's here for use if
// you need it for debugging.
Honey_EXPORT HoneyError HoneyModelContainerRunWithOutputsOnHost(
    HoneyModelHandle handle,
    const HoneyData* inputs,
    size_t num_inputs,
    HoneyData* outputs,
    size_t num_outputs,
    HoneyStreamHandle stream_handle,
    bool graph_mode,
    int64_t** output_shapes_out);

/// Do per op profile and write the profiling report to file.
Honey_EXPORT HoneyError HoneyModelContainerProfile(
    HoneyModelHandle handle,
    const HoneyData* inputs,
    size_t num_inputs,
    HoneyData* outputs,
    size_t num_outputs,
    HoneyStreamHandle stream_handle,
    size_t num_iters,
    const char* filename);

Honey_EXPORT HoneyError HoneyModelContainerBenchmark(
    HoneyModelHandle handle,
    const HoneyData* inputs,
    size_t num_inputs,
    HoneyData* outputs,
    size_t num_outputs,
    HoneyStreamHandle stream_handle,
    bool graph_mode,
    size_t count,
    size_t num_threads,
    bool use_unique_stream_per_thread,
    float* runtime_ms,
    int64_t** output_shapes_out);

Honey_EXPORT HoneyError HoneyModelContainerGetNumInputs(
    HoneyModelHandle handle,
    size_t* num_inputs_out);

Honey_EXPORT HoneyError HoneyModelContainerGetRequiredMemory(
    HoneyModelHandle handle,
    size_t* required_memory);

Honey_EXPORT HoneyError HoneyModelContainerGetInputName(
    HoneyModelHandle handle,
    size_t input_idx,
    const char** input_name_out);

Honey_EXPORT HoneyError HoneyModelContainerGetMaximumInputShape(
    HoneyModelHandle handle,
    size_t input_idx,
    HoneyParamShape* shape);

Honey_EXPORT HoneyError HoneyModelContainerGetInputDtype(
    HoneyModelHandle handle,
    size_t input_idx,
    HoneyDtype* input_dtype);

Honey_EXPORT HoneyError HoneyModelContainerGetNumOutputs(
    HoneyModelHandle handle,
    size_t* num_outputs_out);

Honey_EXPORT HoneyError HoneyModelContainerGetOutputName(
    HoneyModelHandle handle,
    size_t output_idx,
    const char** output_name_out);

Honey_EXPORT HoneyError HoneyModelContainerGetMaximumOutputShape(
    HoneyModelHandle handle,
    size_t output_idx,
    HoneyParamShape* shape_out);

Honey_EXPORT HoneyError HoneyModelContainerGetOutputDtype(
    HoneyModelHandle handle,
    size_t output_idx,
    HoneyDtype* out);

Honey_EXPORT HoneyError HoneyModelContainerGetNumRuntimes(
    HoneyModelHandle handle,
    size_t* num_runtimes_out);

Honey_EXPORT HoneyError HoneyModelContainerFoldConstants(
    HoneyModelHandle handle,
    HoneyStreamHandle stream_handle,
    bool sync);

Honey_EXPORT HoneyError HoneyModelContainerFoldConstantsInDoubleBuffer(
    HoneyModelHandle handle,
    HoneyStreamHandle stream_handle,
    bool sync);

Honey_EXPORT HoneyError
HoneyModelContainerSwapConstants(HoneyModelHandle handle);

Honey_EXPORT HoneyError HoneyAllocatorCreate(
    HoneyAllocator** allocator_out,
    HoneyAllocatorType allocator_type);

Honey_EXPORT HoneyError
HoneyAllocatorDelete(HoneyAllocator* allocator_out);

// Get the number of bytes allocated; mainly used for testing.
Honey_EXPORT HoneyError HoneyTrackingAllocatorGetNumBytes(
    HoneyAllocator* allocator,
    size_t* num_bytes_out);

} // extern "C"
