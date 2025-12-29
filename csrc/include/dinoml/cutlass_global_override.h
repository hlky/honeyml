#pragma once

// CUTLASS override:
// CUTLASS 3.4+ marks kernels static, which breaks cross-TU instantiation.
// We intentionally restore external linkage here.

#include <cutlass/detail/helper_macros.hpp>

#ifdef CUTLASS_GLOBAL
#undef CUTLASS_GLOBAL
#endif

#define CUTLASS_GLOBAL __global__
