#pragma once

#include <cstdint>
#include <type_traits>

#ifdef DINOML_CUDA
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "dinoml/cutlass_global_override.h"

#include "cutlass/util/host_tensor.h"
#endif
#ifdef DINOML_HIP
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include "ck/library/utility/host_tensor.hpp"
#endif

namespace dinoml {
#ifdef DINOML_CUDA
using bfloat16 = __nv_bfloat16;
using bfloat162 = __nv_bfloat162;
using DeviceStream = cudaStream_t;
#define LDG(x) __ldg(x)
#define HALF2DATA(x) x
#endif
#ifdef DINOML_HIP
using bfloat16 = __hip_bfloat16;
using bfloat162 = __hip_bfloat162;
using DeviceStream = hipStream_t;
#define LDG(x) __ldg(x)
#define HALF2DATA(x) x.data
#endif

} // namespace dinoml

#ifdef DINOML_CUDA
using bfloat16 = __nv_bfloat16;
using bfloat162 = __nv_bfloat162;
#endif
#ifdef DINOML_HIP
using bfloat16 = __hip_bfloat16;
using bfloat162 = __hip_bfloat162;
#endif
