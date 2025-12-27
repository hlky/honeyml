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
#include <dinoml/device.h>
#include <stddef.h>
#include <stdint.h>
#include <array>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>
#include "short_file.h"

// We compile all models with -fvisibility=hidden. Any symbols that need to be
// exposed in the final shared library must be declared with DINOML_EXPORT to make
// them visible.

#ifdef __GNUC__ // Applies to any compiler with GNU extensions (clang and g++)
#define DINOML_EXPORT __attribute__((__visibility__("default")))
#else
#ifdef _WIN32
#define DINOML_EXPORT __declspec(dllexport)
#else
#define DINOML_EXPORT
#endif
#endif

struct DinoMLModelOpaque {};
using DinoMLModelHandle = DinoMLModelOpaque*;

enum class DinoMLWorkspaceAllocationMode {
  // workspace is allocated immediately and not released until module is
  // unloaded
  kEager = 0,
  // workspace is allocated at first run and not released until module is
  // unloaded
  kLazy,
  // workspace is allocated each run then freed after use
  kFau
};

enum class DinoMLError : int {
  DinoMLSuccess = 0,
  DinoMLFailure = 1,
};

#define DINOML_ERROR_CHECK(call)                                        \
  if ((call) != DinoMLError::DinoMLSuccess) {                \
    throw std::runtime_error(                                        \
        std::string(#call " API call failed at ") + __SHORT_FILE__ + \
        ", line" + std::to_string(__LINE__));                        \
  }

struct DinoMLParamShape {
  DinoMLParamShape() : shape_data(nullptr), size(0) {}
  DinoMLParamShape(const int64_t* shape_data_in, size_t size_in)
      : shape_data(shape_data_in), size(size_in) {}

  const int64_t* shape_data;
  size_t size;

  size_t Numel() const {
    return std::accumulate(
        shape_data, shape_data + size, (int64_t)1, std::multiplies<int64_t>());
  }
};

enum class DinoMLDtype {
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

struct DinoMLData {
  DinoMLData() : ptr(nullptr), dtype(DinoMLDtype::kUnset) {}

  DinoMLData(
      void* ptr_in,
      const DinoMLParamShape& shape_in,
      DinoMLDtype dtype_in)
      : ptr(ptr_in), shape(shape_in), dtype(dtype_in) {}

  void* ptr;
  DinoMLParamShape shape;
  DinoMLDtype dtype;
};

inline size_t DinoMLDtypeSizeBytes(DinoMLDtype dtype) {
  switch (dtype) {
    case DinoMLDtype::kHalf:
    case DinoMLDtype::kBFloat16:
      return 2;
    case DinoMLDtype::kFloat:
      return 4;
    case DinoMLDtype::kInt:
      return 4;
    case DinoMLDtype::kLong:
      return 8;
    case DinoMLDtype::kFloat8_e4m3:
    case DinoMLDtype::kFloat8_e5m2:
    case DinoMLDtype::kBool:
      return 1;
    case DinoMLDtype::kUnset:
      throw std::runtime_error("Unset dtype has no size!");
  }
  throw std::runtime_error("dtype handling is not implemented!");
}

struct DinoMLStreamOpaque {};
using DinoMLStreamHandle = DinoMLStreamOpaque*;

// Allocator to use for GPU mallocs and frees. Allocations will only happen
// when the ModelContainer is created.
class DinoMLAllocator {
 public:
  virtual void* Allocate(size_t nbytes) = 0;
  virtual void Free(void* ptr) = 0;

  virtual ~DinoMLAllocator() = default;
};

// Some custom allocators are provided. They can be created by passing
// an enum into the DinoMLAllocatorCreate() function.
enum class DinoMLAllocatorType {
  // The default allocator just uses the backend's default malloc/free.
  kDefault = 0,
  // The tracking allocator is like the default allocator, but it keeps
  // track of how many bytes it has allocated. Mainly used for testing.
  kTracking,
};

extern "C" {

DINOML_EXPORT const char* GetLastErrorMessage();

// Create a ModelContainer. See model_container.h for all the details.
// Some important high-level notes:
// * If allocator is null, a default allocator is used (forwards to
//   {cuda/hip}{Malloc/Free}).
// * We assume that the allocator lives at least as long as the ModelContainer.
DINOML_EXPORT DinoMLError DinoMLModelContainerCreate(
    DinoMLModelHandle* ret,
    size_t num_runtimes,
    DinoMLAllocator* allocator = nullptr);

DINOML_EXPORT DinoMLError
DinoMLModelContainerDelete(DinoMLModelHandle handle);

DINOML_EXPORT DinoMLError DinoMLStreamCreate(
  DinoMLStreamHandle* stream_handle,
  bool non_blocking);

DINOML_EXPORT DinoMLError DinoMLModelContainerSetConstant(
    DinoMLModelHandle handle,
    const char* name,
    const DinoMLData* tensor);

DINOML_EXPORT DinoMLError DinoMLModelContainerSetManyConstants(
    DinoMLModelHandle handle,
    const char** names,
    const DinoMLData* tensors,
    size_t num_tensors);

DINOML_EXPORT DinoMLError DinoMLModelContainerSetDoubleBufferConstant(
    DinoMLModelHandle handle,
    DinoMLStreamHandle stream_handle,
    const char* name,
    const DinoMLData* tensor);

DINOML_EXPORT DinoMLError DinoMLModelContainerSetManyDoubleBufferConstants(
    DinoMLModelHandle handle,
    DinoMLStreamHandle stream_handle,
    const char** names,
    const DinoMLData* tensors,
    size_t num_tensors);

DINOML_EXPORT DinoMLError DinoMLModelContainerGetNumConstants(
    DinoMLModelHandle handle,
    bool unbound_constants_only,
    bool constant_folding_inputs_only,
    size_t* num_constants_out);

DINOML_EXPORT DinoMLError DinoMLModelContainerGetConstantNames(
    DinoMLModelHandle handle,
    bool unbound_constants_only,
    bool constant_folding_inputs_only,
    const char** constant_names_out);

DINOML_EXPORT DinoMLError DinoMLModelContainerGetConstantDtype(
    DinoMLModelHandle handle,
    const char* name,
    DinoMLDtype* dtype);

DINOML_EXPORT DinoMLError DinoMLModelContainerGetConstantOriginalName(
    DinoMLModelHandle handle,
    const char* name,
    const char** original_name_out);

DINOML_EXPORT DinoMLError DinoMLModelContainerRun(
    DinoMLModelHandle handle,
    const DinoMLData* inputs,
    size_t num_inputs,
    DinoMLData* outputs,
    size_t num_outputs,
    DinoMLStreamHandle stream_handle,
    bool sync,
    bool graph_mode,
    int64_t** output_shapes_out);

// Like DinoMLModelContainerRun, but expects outputs to be allocated on the
// host. Does an extra sync/copy at the end to copy them over. Warning: don't
// use this! It's not optimal with respect to performance. It's here for use if
// you need it for debugging.
DINOML_EXPORT DinoMLError DinoMLModelContainerRunWithOutputsOnHost(
    DinoMLModelHandle handle,
    const DinoMLData* inputs,
    size_t num_inputs,
    DinoMLData* outputs,
    size_t num_outputs,
    DinoMLStreamHandle stream_handle,
    bool graph_mode,
    int64_t** output_shapes_out);

/// Do per op profile and write the profiling report to file.
DINOML_EXPORT DinoMLError DinoMLModelContainerProfile(
    DinoMLModelHandle handle,
    const DinoMLData* inputs,
    size_t num_inputs,
    DinoMLData* outputs,
    size_t num_outputs,
    DinoMLStreamHandle stream_handle,
    size_t num_iters,
    const char* filename);

DINOML_EXPORT DinoMLError DinoMLModelContainerBenchmark(
    DinoMLModelHandle handle,
    const DinoMLData* inputs,
    size_t num_inputs,
    DinoMLData* outputs,
    size_t num_outputs,
    DinoMLStreamHandle stream_handle,
    bool graph_mode,
    size_t count,
    size_t num_threads,
    bool use_unique_stream_per_thread,
    float* runtime_ms,
    int64_t** output_shapes_out);

DINOML_EXPORT DinoMLError DinoMLModelContainerGetNumInputs(
    DinoMLModelHandle handle,
    size_t* num_inputs_out);

DINOML_EXPORT DinoMLError DinoMLModelContainerGetRequiredMemory(
    DinoMLModelHandle handle,
    size_t* required_memory);

DINOML_EXPORT DinoMLError DinoMLModelContainerGetInputName(
    DinoMLModelHandle handle,
    size_t input_idx,
    const char** input_name_out);

DINOML_EXPORT DinoMLError DinoMLModelContainerGetMaximumInputShape(
    DinoMLModelHandle handle,
    size_t input_idx,
    DinoMLParamShape* shape);

DINOML_EXPORT DinoMLError DinoMLModelContainerGetInputDtype(
    DinoMLModelHandle handle,
    size_t input_idx,
    DinoMLDtype* input_dtype);

DINOML_EXPORT DinoMLError DinoMLModelContainerGetNumOutputs(
    DinoMLModelHandle handle,
    size_t* num_outputs_out);

DINOML_EXPORT DinoMLError DinoMLModelContainerGetOutputName(
    DinoMLModelHandle handle,
    size_t output_idx,
    const char** output_name_out);

DINOML_EXPORT DinoMLError DinoMLModelContainerGetMaximumOutputShape(
    DinoMLModelHandle handle,
    size_t output_idx,
    DinoMLParamShape* shape_out);

DINOML_EXPORT DinoMLError DinoMLModelContainerGetOutputDtype(
    DinoMLModelHandle handle,
    size_t output_idx,
    DinoMLDtype* out);

DINOML_EXPORT DinoMLError DinoMLModelContainerGetNumRuntimes(
    DinoMLModelHandle handle,
    size_t* num_runtimes_out);

DINOML_EXPORT DinoMLError DinoMLModelContainerFoldConstants(
    DinoMLModelHandle handle,
    DinoMLStreamHandle stream_handle,
    bool sync);

DINOML_EXPORT DinoMLError DinoMLModelContainerFoldConstantsInDoubleBuffer(
    DinoMLModelHandle handle,
    DinoMLStreamHandle stream_handle,
    bool sync);

DINOML_EXPORT DinoMLError
DinoMLModelContainerSwapConstants(DinoMLModelHandle handle);

DINOML_EXPORT DinoMLError DinoMLAllocatorCreate(
    DinoMLAllocator** allocator_out,
    DinoMLAllocatorType allocator_type);

DINOML_EXPORT DinoMLError
DinoMLAllocatorDelete(DinoMLAllocator* allocator_out);

// Get the number of bytes allocated; mainly used for testing.
DINOML_EXPORT DinoMLError DinoMLTrackingAllocatorGetNumBytes(
    DinoMLAllocator* allocator,
    size_t* num_bytes_out);

} // extern "C"
