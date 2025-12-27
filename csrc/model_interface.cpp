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
    return DinoMLError::DinoMLFailure;   \
  } catch (...) {                                \
    last_error_message = "Unknown exception occurred."; \
    LOG(ERROR) << "Unknown exception occurred."; \
    return DinoMLError::DinoMLFailure;   \
  }                                              \
  return DinoMLError::DinoMLSuccess;

#define RETURN_ERROR_IF_NULL(var)                          \
  if (var == nullptr) {                                    \
    last_error_message = "Variable " #var " can't be null";\
    LOG(ERROR) << "Variable " << #var << " can't be null"; \
    return DinoMLError::DinoMLFailure;             \
  }

namespace dinoml {
namespace {
class DefaultAllocator : public DinoMLAllocator {
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
} // namespace dinoml

extern "C" {

const char* GetLastErrorMessage() {
  return last_error_message.c_str();
}

DinoMLError DinoMLModelContainerCreate(
    DinoMLModelHandle* ret,
    size_t num_runtimes,
    DinoMLAllocator* allocator) {
  if (num_runtimes == 0) {
    last_error_message = "num_runtimes must be positive, but got 0";
    LOG(ERROR) << "num_runtimes must be positive, but got 0";
    return DinoMLError::DinoMLFailure;
  }
  RETURN_ERROR_IF_NULL(ret)
  DinoMLAllocator& allocator_ref =
      allocator == nullptr ? dinoml::default_allocator : *allocator;
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* m = dinoml::CreateModelContainer(num_runtimes, allocator_ref);
    *ret = reinterpret_cast<DinoMLModelHandle>(m);
  })
}

DinoMLError DinoMLModelContainerDelete(DinoMLModelHandle handle) {
  RETURN_ERROR_IF_NULL(handle)
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* m = reinterpret_cast<dinoml::ModelContainer*>(handle);
    delete m;
  });
}

DinoMLError DinoMLStreamCreate(
  DinoMLStreamHandle* handle,
  bool non_blocking) {
  RETURN_ERROR_IF_NULL(handle)
  auto stream = dinoml::RAII_StreamCreate(non_blocking);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *handle = reinterpret_cast<DinoMLStreamHandle>(stream.get()); });
}

DinoMLError DinoMLModelContainerSetConstant(
    DinoMLModelHandle handle,
    const char* name,
    const DinoMLData* tensor) {
  RETURN_ERROR_IF_NULL(handle)
  RETURN_ERROR_IF_NULL(tensor)
  auto* m = reinterpret_cast<dinoml::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ m->SetConstant(name, *tensor); })
}

DINOML_EXPORT DinoMLError DinoMLModelContainerSetManyConstants(
    DinoMLModelHandle handle,
    const char** names,
    const DinoMLData* tensors,
    size_t num_tensors) {
  RETURN_ERROR_IF_NULL(handle)
  auto* m = reinterpret_cast<dinoml::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { m->SetManyConstants(names, tensors, num_tensors); })
}

DinoMLError DinoMLModelContainerSetDoubleBufferConstant(
    DinoMLModelHandle handle,
    DinoMLStreamHandle stream_handle,
    const char* name,
    const DinoMLData* tensor) {
  RETURN_ERROR_IF_NULL(handle)
  RETURN_ERROR_IF_NULL(tensor)
  auto* m = reinterpret_cast<dinoml::ModelContainer*>(handle);
  auto stream = reinterpret_cast<dinoml::StreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { m->SetDoubleBufferConstant(name, *tensor, stream); })
}

DINOML_EXPORT DinoMLError DinoMLModelContainerSetManyDoubleBufferConstants(
    DinoMLModelHandle handle,
    DinoMLStreamHandle stream_handle,
    const char** names,
    const DinoMLData* tensors,
    size_t num_tensors) {
  RETURN_ERROR_IF_NULL(handle)
  auto* m = reinterpret_cast<dinoml::ModelContainer*>(handle);
  auto stream = reinterpret_cast<dinoml::StreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { m->SetManyDoubleBufferConstants(names, tensors, num_tensors, stream); })
}

DinoMLError DinoMLModelContainerGetNumConstants(
    DinoMLModelHandle handle,
    bool unbound_constants_only,
    bool constant_folding_inputs_only,
    size_t* num_constants_out) {
  RETURN_ERROR_IF_NULL(handle)
  RETURN_ERROR_IF_NULL(num_constants_out)
  auto* m = reinterpret_cast<dinoml::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    if (constant_folding_inputs_only) {
      *num_constants_out =
          m->GetNumConstantFoldingInputs(unbound_constants_only);
    } else {
      *num_constants_out = m->GetNumConstants(unbound_constants_only);
    }
  })
}

DinoMLError DinoMLModelContainerGetConstantNames(
    DinoMLModelHandle handle,
    bool unbound_constants_only,
    bool constant_folding_inputs_only,
    const char** constant_names_out) {
  RETURN_ERROR_IF_NULL(handle)
  // WriteAllConstantNamesTo() will handle nullptr checks on constant_names_out.
  // Passing nullptr is allowed if there are 0 constants!
  auto* m = reinterpret_cast<dinoml::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    m->WriteAllConstantNamesTo(
        constant_names_out,
        unbound_constants_only,
        constant_folding_inputs_only);
  })
}

DinoMLError DinoMLModelContainerGetConstantDtype(
    DinoMLModelHandle handle,
    const char* name,
    DinoMLDtype* dtype) {
  RETURN_ERROR_IF_NULL(handle)
  RETURN_ERROR_IF_NULL(dtype)
  auto* m = reinterpret_cast<dinoml::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *dtype = m->ConstantDtype(name); })
}

DinoMLError DinoMLModelContainerGetConstantOriginalName(
    DinoMLModelHandle handle,
    const char* name,
    const char** original_name_out) {
  RETURN_ERROR_IF_NULL(handle)
  RETURN_ERROR_IF_NULL(original_name_out)
  auto* m = reinterpret_cast<dinoml::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *original_name_out = m->ConstantOriginalName(name); })
}

DinoMLError DinoMLModelContainerRun(
    DinoMLModelHandle handle,
    const DinoMLData* inputs,
    size_t num_inputs,
    DinoMLData* outputs,
    size_t num_outputs,
    DinoMLStreamHandle stream_handle,
    bool sync,
    bool graph_mode,
    int64_t** output_shapes_out) {
  RETURN_ERROR_IF_NULL(handle)
  auto* m = reinterpret_cast<dinoml::ModelContainer*>(handle);
  auto stream = reinterpret_cast<dinoml::StreamType>(stream_handle);
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

DinoMLError DinoMLModelContainerRunWithOutputsOnHost(
    DinoMLModelHandle handle,
    const DinoMLData* inputs,
    size_t num_inputs,
    DinoMLData* outputs,
    size_t num_outputs,
    DinoMLStreamHandle stream_handle,
    bool graph_mode,
    int64_t** output_shapes_out) {
  RETURN_ERROR_IF_NULL(handle)
  auto* m = reinterpret_cast<dinoml::ModelContainer*>(handle);
  auto stream = reinterpret_cast<dinoml::StreamType>(stream_handle);
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

DinoMLError DinoMLModelContainerProfile(
    DinoMLModelHandle handle,
    const DinoMLData* inputs,
    size_t num_inputs,
    DinoMLData* outputs,
    size_t num_outputs,
    DinoMLStreamHandle stream_handle,
    size_t num_iters,
    const char* filename) {
  RETURN_ERROR_IF_NULL(handle);
  RETURN_ERROR_IF_NULL(filename);
  auto* m = reinterpret_cast<dinoml::ModelContainer*>(handle);
  auto stream = reinterpret_cast<dinoml::StreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    m->Profile(
        inputs, num_inputs, outputs, num_outputs, stream, num_iters, filename);
  })
}

DinoMLError DinoMLModelContainerBenchmark(
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
    int64_t** output_shapes_out) {
  RETURN_ERROR_IF_NULL(handle)
  RETURN_ERROR_IF_NULL(runtime_ms)
  auto* m = reinterpret_cast<dinoml::ModelContainer*>(handle);
  auto stream = reinterpret_cast<dinoml::StreamType>(stream_handle);
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

DinoMLError DinoMLModelContainerGetNumInputs(
    DinoMLModelHandle handle,
    size_t* num_inputs_out) {
  RETURN_ERROR_IF_NULL(handle)
  RETURN_ERROR_IF_NULL(num_inputs_out)
  auto* m = reinterpret_cast<dinoml::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *num_inputs_out = m->NumInputs(); })
}

DinoMLError DinoMLModelContainerGetRequiredMemory(
  DinoMLModelHandle handle,
  size_t* required_memory) {
RETURN_ERROR_IF_NULL(handle)
RETURN_ERROR_IF_NULL(required_memory)
auto* m = reinterpret_cast<dinoml::ModelContainer*>(handle);
CONVERT_EXCEPTION_TO_ERROR_CODE({ *required_memory = m->RequiredMemory(); })
}

DinoMLError DinoMLModelContainerGetInputName(
    DinoMLModelHandle handle,
    size_t input_idx,
    const char** input_name_out) {
  RETURN_ERROR_IF_NULL(handle)
  RETURN_ERROR_IF_NULL(input_name_out)
  auto* m = reinterpret_cast<dinoml::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *input_name_out = m->InputName(input_idx); })
}

DinoMLError DinoMLModelContainerGetMaximumInputShape(
    DinoMLModelHandle handle,
    size_t input_idx,
    DinoMLParamShape* shape) {
  RETURN_ERROR_IF_NULL(handle)
  RETURN_ERROR_IF_NULL(shape)
  auto* m = reinterpret_cast<dinoml::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *shape = m->MaxInputShape(input_idx); })
}

DinoMLError DinoMLModelContainerGetInputDtype(
    DinoMLModelHandle handle,
    size_t input_idx,
    DinoMLDtype* input_dtype) {
  RETURN_ERROR_IF_NULL(handle)
  RETURN_ERROR_IF_NULL(input_dtype)
  auto* m = reinterpret_cast<dinoml::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *input_dtype = m->InputDtype(input_idx); })
}

DinoMLError DinoMLModelContainerGetNumOutputs(
    DinoMLModelHandle handle,
    size_t* num_outputs_out) {
  RETURN_ERROR_IF_NULL(handle)
  RETURN_ERROR_IF_NULL(num_outputs_out)
  auto* m = reinterpret_cast<dinoml::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *num_outputs_out = m->NumOutputs(); })
}

DinoMLError DinoMLModelContainerGetOutputName(
    DinoMLModelHandle handle,
    size_t output_idx,
    const char** output_name_out) {
  RETURN_ERROR_IF_NULL(handle)
  RETURN_ERROR_IF_NULL(output_name_out)
  auto* m = reinterpret_cast<dinoml::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *output_name_out = m->OutputName(output_idx); })
}

DinoMLError DinoMLModelContainerGetMaximumOutputShape(
    DinoMLModelHandle handle,
    size_t output_idx,
    DinoMLParamShape* shape_out) {
  RETURN_ERROR_IF_NULL(handle)
  RETURN_ERROR_IF_NULL(shape_out)
  auto* m = reinterpret_cast<dinoml::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *shape_out = m->MaxOutputShape(output_idx); })
}

DinoMLError DinoMLModelContainerGetOutputDtype(
    DinoMLModelHandle handle,
    size_t output_idx,
    DinoMLDtype* dtype_out) {
  RETURN_ERROR_IF_NULL(handle)
  RETURN_ERROR_IF_NULL(dtype_out)
  auto* m = reinterpret_cast<dinoml::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *dtype_out = m->OutputDtype(output_idx); })
}

DinoMLError DinoMLModelContainerGetNumRuntimes(
    DinoMLModelHandle handle,
    size_t* num_runtimes_out) {
  RETURN_ERROR_IF_NULL(num_runtimes_out)
  auto* m = reinterpret_cast<dinoml::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *num_runtimes_out = m->GetNumRuntimes(); })
}

DinoMLError DinoMLModelContainerFoldConstants(
    DinoMLModelHandle handle,
    DinoMLStreamHandle stream_handle,
    bool sync) {
  RETURN_ERROR_IF_NULL(handle)
  auto* m = reinterpret_cast<dinoml::ModelContainer*>(handle);
  auto stream = reinterpret_cast<dinoml::StreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ m->FoldConstants(stream, sync, false); })
}

DinoMLError DinoMLModelContainerFoldConstantsInDoubleBuffer(
    DinoMLModelHandle handle,
    DinoMLStreamHandle stream_handle,
    bool sync) {
  RETURN_ERROR_IF_NULL(handle)
  auto* m = reinterpret_cast<dinoml::ModelContainer*>(handle);
  auto stream = reinterpret_cast<dinoml::StreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ m->FoldConstants(stream, sync, true); })
}

DinoMLError DinoMLModelContainerSwapConstants(
    DinoMLModelHandle handle) {
  RETURN_ERROR_IF_NULL(handle)
  auto* m = reinterpret_cast<dinoml::ModelContainer*>(handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ m->SwapConstants(); })
}

DinoMLError DinoMLAllocatorCreate(
    DinoMLAllocator** allocator_out,
    DinoMLAllocatorType allocator_type) {
  RETURN_ERROR_IF_NULL(allocator_out);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    switch (allocator_type) {
      case DinoMLAllocatorType::kDefault:
        *allocator_out = new dinoml::DefaultAllocator();
        break;
      case DinoMLAllocatorType::kTracking:
        *allocator_out = new dinoml::TrackingAllocator();
        break;
      default:
        throw std::runtime_error("Unrecognized allocator type");
    }
  });
}

DinoMLError DinoMLAllocatorDelete(DinoMLAllocator* allocator) {
  RETURN_ERROR_IF_NULL(allocator);
  delete allocator;
  return DinoMLError::DinoMLSuccess;
}

DinoMLError DinoMLTrackingAllocatorGetNumBytes(
    DinoMLAllocator* allocator,
    size_t* num_bytes_out) {
  RETURN_ERROR_IF_NULL(allocator);
  RETURN_ERROR_IF_NULL(num_bytes_out);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* tracking_allocator = dynamic_cast<dinoml::TrackingAllocator*>(allocator);
    if (tracking_allocator == nullptr) {
      throw std::runtime_error("Allocator was not a tracking allocator!");
    }
    *num_bytes_out = tracking_allocator->NumBytesAllocated();
  });
}

} // extern "C"
