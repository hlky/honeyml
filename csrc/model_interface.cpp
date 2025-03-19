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
#include "model_interface.h"
#include <iostream>
#include <unordered_map>
#include "model-generated.h"
#include "model_container.h"
#include "raii_wrapper.h"
#include <string>
#include <thread>

thread_local std::string last_error_message;

// Important: don't let exceptions escape the functions below.
// They can cause problems when -fvisibility=hidden. But more
// importantly, they can crash the program if they try to cross
// the language boundary into Python.

#define CONVERT_EXCEPTION_TO_ERROR_CODE(...)     \
  try {                                          \
    __VA_ARGS__                                  \
  } catch (const std::exception& e) {            \
    last_error_message = e.what();               \
    LOG(ERROR) << "Error: " << e.what();         \
    return HoneyError::HoneyFailure;   \
  } catch (...) {                                \
    last_error_message = "Unknown exception occurred."; \
    LOG(ERROR) << "Unknown exception occurred."; \
    return HoneyError::HoneyFailure;   \
  }                                              \
  return HoneyError::HoneySuccess;

#define RETURN_ERROR_IF_NULL(var)                          \
  if (var == nullptr) {                                    \
    last_error_message = "Variable " #var " can't be null";\
    LOG(ERROR) << "Variable " << #var << " can't be null"; \
    return HoneyError::HoneyFailure;             \
  }

namespace honey {
namespace {
class DefaultAllocator : public HoneyAllocator {
 public:
  void* Allocate(size_t n_bytes) override {
    void* result;
    DEVICE_CHECK(DeviceMalloc(&result, n_bytes));
    return result;
  }

  void Free(void* ptr) override {
    DEVICE_CHECK(FreeDeviceMemory(ptr));
  }
};

class TrackingAllocator : public DefaultAllocator {
 public:
  void* Allocate(size_t n_bytes) override {
    auto* result = DefaultAllocator::Allocate(n_bytes);
    num_bytes_ += n_bytes;
    return result;
  }

  size_t NumBytesAllocated() const {
    return num_bytes_;
  }

 private:
  size_t num_bytes_ = 0;
};

DefaultAllocator default_allocator;
} // namespace
} // namespace honey

extern "C" {

const char* GetLastErrorMessage() {
  return last_error_message.c_str();
}

HoneyError HoneyModelContainerCreate(
    HoneyModelHandle* ret,
    size_t num_runtimes,
    HoneyAllocator* allocator) {
  if (num_runtimes == 0) {
    last_error_message = "num_runtimes must be positive, but got 0";
    LOG(ERROR) << "num_runtimes must be positive, but got 0";
    return HoneyError::HoneyFailure;
  }
  RETURN_ERROR_IF_NULL(ret)
  HoneyAllocator& allocator_ref =
      allocator == nullptr ? honey::default_allocator : *allocator;
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* m = honey::CreateModelContainer(num_runtimes, allocator_ref);
    *ret = reinterpret_cast<HoneyModelHandle>(m);
  })
}

HoneyError HoneyModelContainerDelete(HoneyModelHandle handle) {
  RETURN_ERROR_IF_NULL(handle)
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* m = reinterpret_cast<honey::ModelContainer*>(handle);
    delete m;
  });
}

HoneyError HoneyStreamCreate(
  HoneyStreamHandle* handle,
  bool non_blocking) {
  RETURN_ERROR_IF_NULL(handle)
  auto stream = honey::RAII_StreamCreate(non_blocking);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *handle = reinterpret_cast<HoneyStreamHandle>(stream.get()); });
}

HoneyError HoneyModelContainerSetConstant(
    HoneyModelHandle handle,
    const char* name,
    const HoneyData* tensor) {
  RETURN_ERROR_IF_NULL(handle)
  RETURN_ERROR_IF_NULL(tensor)
  auto* m = reinterpret_cast<honey::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ m->SetConstant(name, *tensor); })
}

Honey_EXPORT HoneyError HoneyModelContainerSetManyConstants(
    HoneyModelHandle handle,
    const char** names,
    const HoneyData* tensors,
    size_t num_tensors) {
  RETURN_ERROR_IF_NULL(handle)
  auto* m = reinterpret_cast<honey::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { m->SetManyConstants(names, tensors, num_tensors); })
}

HoneyError HoneyModelContainerSetDoubleBufferConstant(
    HoneyModelHandle handle,
    HoneyStreamHandle stream_handle,
    const char* name,
    const HoneyData* tensor) {
  RETURN_ERROR_IF_NULL(handle)
  RETURN_ERROR_IF_NULL(tensor)
  auto* m = reinterpret_cast<honey::ModelContainer*>(handle);
  auto stream = reinterpret_cast<honey::StreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { m->SetDoubleBufferConstant(name, *tensor, stream); })
}

Honey_EXPORT HoneyError HoneyModelContainerSetManyDoubleBufferConstants(
    HoneyModelHandle handle,
    HoneyStreamHandle stream_handle,
    const char** names,
    const HoneyData* tensors,
    size_t num_tensors) {
  RETURN_ERROR_IF_NULL(handle)
  auto* m = reinterpret_cast<honey::ModelContainer*>(handle);
  auto stream = reinterpret_cast<honey::StreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { m->SetManyDoubleBufferConstants(names, tensors, num_tensors, stream); })
}

HoneyError HoneyModelContainerGetNumConstants(
    HoneyModelHandle handle,
    bool unbound_constants_only,
    bool constant_folding_inputs_only,
    size_t* num_constants_out) {
  RETURN_ERROR_IF_NULL(handle)
  RETURN_ERROR_IF_NULL(num_constants_out)
  auto* m = reinterpret_cast<honey::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    if (constant_folding_inputs_only) {
      *num_constants_out =
          m->GetNumConstantFoldingInputs(unbound_constants_only);
    } else {
      *num_constants_out = m->GetNumConstants(unbound_constants_only);
    }
  })
}

HoneyError HoneyModelContainerGetConstantNames(
    HoneyModelHandle handle,
    bool unbound_constants_only,
    bool constant_folding_inputs_only,
    const char** constant_names_out) {
  RETURN_ERROR_IF_NULL(handle)
  // WriteAllConstantNamesTo() will handle nullptr checks on constant_names_out.
  // Passing nullptr is allowed if there are 0 constants!
  auto* m = reinterpret_cast<honey::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    m->WriteAllConstantNamesTo(
        constant_names_out,
        unbound_constants_only,
        constant_folding_inputs_only);
  })
}

HoneyError HoneyModelContainerGetConstantDtype(
    HoneyModelHandle handle,
    const char* name,
    HoneyDtype* dtype) {
  RETURN_ERROR_IF_NULL(handle)
  RETURN_ERROR_IF_NULL(dtype)
  auto* m = reinterpret_cast<honey::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *dtype = m->ConstantDtype(name); })
}

HoneyError HoneyModelContainerGetConstantOriginalName(
    HoneyModelHandle handle,
    const char* name,
    const char** original_name_out) {
  RETURN_ERROR_IF_NULL(handle)
  RETURN_ERROR_IF_NULL(original_name_out)
  auto* m = reinterpret_cast<honey::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *original_name_out = m->ConstantOriginalName(name); })
}

HoneyError HoneyModelContainerRun(
    HoneyModelHandle handle,
    const HoneyData* inputs,
    size_t num_inputs,
    HoneyData* outputs,
    size_t num_outputs,
    HoneyStreamHandle stream_handle,
    bool sync,
    bool graph_mode,
    int64_t** output_shapes_out) {
  RETURN_ERROR_IF_NULL(handle)
  auto* m = reinterpret_cast<honey::ModelContainer*>(handle);
  auto stream = reinterpret_cast<honey::StreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    m->Run(
        inputs,
        num_inputs,
        outputs,
        num_outputs,
        stream,
        sync,
        graph_mode,
        output_shapes_out);
  })
}

HoneyError HoneyModelContainerRunWithOutputsOnHost(
    HoneyModelHandle handle,
    const HoneyData* inputs,
    size_t num_inputs,
    HoneyData* outputs,
    size_t num_outputs,
    HoneyStreamHandle stream_handle,
    bool graph_mode,
    int64_t** output_shapes_out) {
  RETURN_ERROR_IF_NULL(handle)
  auto* m = reinterpret_cast<honey::ModelContainer*>(handle);
  auto stream = reinterpret_cast<honey::StreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    m->RunWithOutputsOnHost(
        inputs,
        num_inputs,
        outputs,
        num_outputs,
        stream,
        graph_mode,
        output_shapes_out);
  })
}

HoneyError HoneyModelContainerProfile(
    HoneyModelHandle handle,
    const HoneyData* inputs,
    size_t num_inputs,
    HoneyData* outputs,
    size_t num_outputs,
    HoneyStreamHandle stream_handle,
    size_t num_iters,
    const char* filename) {
  RETURN_ERROR_IF_NULL(handle);
  RETURN_ERROR_IF_NULL(filename);
  auto* m = reinterpret_cast<honey::ModelContainer*>(handle);
  auto stream = reinterpret_cast<honey::StreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    m->Profile(
        inputs, num_inputs, outputs, num_outputs, stream, num_iters, filename);
  })
}

HoneyError HoneyModelContainerBenchmark(
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
    int64_t** output_shapes_out) {
  RETURN_ERROR_IF_NULL(handle)
  RETURN_ERROR_IF_NULL(runtime_ms)
  auto* m = reinterpret_cast<honey::ModelContainer*>(handle);
  auto stream = reinterpret_cast<honey::StreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    *runtime_ms = m->Benchmark(
        inputs,
        num_inputs,
        outputs,
        num_outputs,
        stream,
        graph_mode,
        count,
        num_threads,
        use_unique_stream_per_thread,
        output_shapes_out);
  })
}

HoneyError HoneyModelContainerGetNumInputs(
    HoneyModelHandle handle,
    size_t* num_inputs_out) {
  RETURN_ERROR_IF_NULL(handle)
  RETURN_ERROR_IF_NULL(num_inputs_out)
  auto* m = reinterpret_cast<honey::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *num_inputs_out = m->NumInputs(); })
}

HoneyError HoneyModelContainerGetRequiredMemory(
  HoneyModelHandle handle,
  size_t* required_memory) {
RETURN_ERROR_IF_NULL(handle)
RETURN_ERROR_IF_NULL(required_memory)
auto* m = reinterpret_cast<honey::ModelContainer*>(handle);
CONVERT_EXCEPTION_TO_ERROR_CODE({ *required_memory = m->RequiredMemory(); })
}

HoneyError HoneyModelContainerGetInputName(
    HoneyModelHandle handle,
    size_t input_idx,
    const char** input_name_out) {
  RETURN_ERROR_IF_NULL(handle)
  RETURN_ERROR_IF_NULL(input_name_out)
  auto* m = reinterpret_cast<honey::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *input_name_out = m->InputName(input_idx); })
}

HoneyError HoneyModelContainerGetMaximumInputShape(
    HoneyModelHandle handle,
    size_t input_idx,
    HoneyParamShape* shape) {
  RETURN_ERROR_IF_NULL(handle)
  RETURN_ERROR_IF_NULL(shape)
  auto* m = reinterpret_cast<honey::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *shape = m->MaxInputShape(input_idx); })
}

HoneyError HoneyModelContainerGetInputDtype(
    HoneyModelHandle handle,
    size_t input_idx,
    HoneyDtype* input_dtype) {
  RETURN_ERROR_IF_NULL(handle)
  RETURN_ERROR_IF_NULL(input_dtype)
  auto* m = reinterpret_cast<honey::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *input_dtype = m->InputDtype(input_idx); })
}

HoneyError HoneyModelContainerGetNumOutputs(
    HoneyModelHandle handle,
    size_t* num_outputs_out) {
  RETURN_ERROR_IF_NULL(handle)
  RETURN_ERROR_IF_NULL(num_outputs_out)
  auto* m = reinterpret_cast<honey::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *num_outputs_out = m->NumOutputs(); })
}

HoneyError HoneyModelContainerGetOutputName(
    HoneyModelHandle handle,
    size_t output_idx,
    const char** output_name_out) {
  RETURN_ERROR_IF_NULL(handle)
  RETURN_ERROR_IF_NULL(output_name_out)
  auto* m = reinterpret_cast<honey::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *output_name_out = m->OutputName(output_idx); })
}

HoneyError HoneyModelContainerGetMaximumOutputShape(
    HoneyModelHandle handle,
    size_t output_idx,
    HoneyParamShape* shape_out) {
  RETURN_ERROR_IF_NULL(handle)
  RETURN_ERROR_IF_NULL(shape_out)
  auto* m = reinterpret_cast<honey::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *shape_out = m->MaxOutputShape(output_idx); })
}

HoneyError HoneyModelContainerGetOutputDtype(
    HoneyModelHandle handle,
    size_t output_idx,
    HoneyDtype* dtype_out) {
  RETURN_ERROR_IF_NULL(handle)
  RETURN_ERROR_IF_NULL(dtype_out)
  auto* m = reinterpret_cast<honey::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *dtype_out = m->OutputDtype(output_idx); })
}

HoneyError HoneyModelContainerGetNumRuntimes(
    HoneyModelHandle handle,
    size_t* num_runtimes_out) {
  RETURN_ERROR_IF_NULL(num_runtimes_out)
  auto* m = reinterpret_cast<honey::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *num_runtimes_out = m->GetNumRuntimes(); })
}

HoneyError HoneyModelContainerFoldConstants(
    HoneyModelHandle handle,
    HoneyStreamHandle stream_handle,
    bool sync) {
  RETURN_ERROR_IF_NULL(handle)
  auto* m = reinterpret_cast<honey::ModelContainer*>(handle);
  auto stream = reinterpret_cast<honey::StreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ m->FoldConstants(stream, sync, false); })
}

HoneyError HoneyModelContainerFoldConstantsInDoubleBuffer(
    HoneyModelHandle handle,
    HoneyStreamHandle stream_handle,
    bool sync) {
  RETURN_ERROR_IF_NULL(handle)
  auto* m = reinterpret_cast<honey::ModelContainer*>(handle);
  auto stream = reinterpret_cast<honey::StreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ m->FoldConstants(stream, sync, true); })
}

HoneyError HoneyModelContainerSwapConstants(
    HoneyModelHandle handle) {
  RETURN_ERROR_IF_NULL(handle)
  auto* m = reinterpret_cast<honey::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ m->SwapConstants(); })
}

HoneyError HoneyAllocatorCreate(
    HoneyAllocator** allocator_out,
    HoneyAllocatorType allocator_type) {
  RETURN_ERROR_IF_NULL(allocator_out);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    switch (allocator_type) {
      case HoneyAllocatorType::kDefault:
        *allocator_out = new honey::DefaultAllocator();
        break;
      case HoneyAllocatorType::kTracking:
        *allocator_out = new honey::TrackingAllocator();
        break;
      default:
        throw std::runtime_error("Unrecognized allocator type");
    }
  });
}

HoneyError HoneyAllocatorDelete(HoneyAllocator* allocator) {
  RETURN_ERROR_IF_NULL(allocator);
  delete allocator;
  return HoneyError::HoneySuccess;
}

HoneyError HoneyTrackingAllocatorGetNumBytes(
    HoneyAllocator* allocator,
    size_t* num_bytes_out) {
  RETURN_ERROR_IF_NULL(allocator);
  RETURN_ERROR_IF_NULL(num_bytes_out);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* tracking_allocator = dynamic_cast<honey::TrackingAllocator*>(allocator);
    if (tracking_allocator == nullptr) {
      throw std::runtime_error("Allocator was not a tracking allocator!");
    }
    *num_bytes_out = tracking_allocator->NumBytesAllocated();
  });
}

} // extern "C"
