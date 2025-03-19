#pragma once
#ifdef Honey_CUDA
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "cutlass/util/host_tensor.h"
#endif
#ifdef Honey_HIP
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include "library/include/ck/library/utility/host_tensor.hpp"
#endif

namespace honey {
#ifdef Honey_CUDA
using bfloat16 = __nv_bfloat16;
using DeviceStream = cudaStream_t;
#endif
#ifdef Honey_HIP
using DeviceStream = hipStream_t;
#endif
} // namespace honey