#pragma once
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
using DeviceStream = hipStream_t;
#endif
} // namespace dinoml