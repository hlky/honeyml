# Environment

- `DINOML_COMPILER_OPT`
- `DINOML_USE_FAST_MATH`
- `DINOML_USE_TANH_FOR_SIGMOID`
- `DINOML_ENABLE_CUDA_LTO`
- `DINOML_NVCC_CCBIN`
- `DINOML_FORCE_PROFILER_CACHE`
- `DINOML_TIME_COMPILATION`
- `DINOML_PLOT_SHORTEN_TENSOR_NAMES`
- `DINOML_BUILD_CACHE_DIR`
- `DINOML_BUILD_CACHE_SKIP_PERCENTAGE`
- `DINOML_BUILD_CACHE_SKIP_PROFILER`
- `DINOML_BUILD_CACHE_MAX_MB`
- `DINOML_ALLOW_CUTLASS_SM90_KERNELS`
- `DINOML_FORCE_CUTLASS_SM90_KERNELS`
- `DINOML_MULTISTREAM_MODE`
- `DINOML_MULTISTREAM_EXTRA_STREAMS`
- `DINOML_MULTISTREAM_MAX_MEM_PARALLEL_OPS`
- `DINOML_ALLOCATION_MODE`
- `DINOML_USE_CMAKE_COMPILATION`
- `DINOML_ENABLE_STANDALONE`
- `DINOML_ENABLE_PTXAS_INFO`
- `DINOML_CUDA_DEBUG_LEVEL`
- `DINOML_NDEBUG`
- `CUTLASS_DEBUG_TRACE_LEVEL`

## Profiling Notes

The list above only covers the main `DINOML_*` knobs. Profiling behavior also depends on a few additional environment variables used by the target and cache layers.

- `DINOML_FORCE_PROFILER_CACHE=1`
  Fail on a compile-time profiling cache miss instead of launching profilers.
- `FORCE_PROFILE=1`
  Force live profiling even in CI-like environments.
- `DISABLE_PROFILER_CODEGEN=1`
  Skip profiler code generation unless `FORCE_PROFILE=1` is set.
- `CACHE_DIR=/path/to/cache`
  Override the profile cache directory. By default DinoML uses `~/.dinoml/`.
- `FLUSH_PROFILE_CACHE=1`
  Delete the target cache DB before loading it.
- `CI_FLAG=CIRCLECI`
  Enables CI-specific target behavior, including dummy profiling results when applicable.
- `TRICK_CI_ENV=1`
  Override CI detection and restore normal profiling behavior.

For the full profiling flow, cache behavior, and runtime profiling APIs, see [PROFILING](./PROFILING.md).

## Target kwargs

- `elementwise_use_fp32_acc`
- `use_fp16_acc`
- `use_fast_math`
- `use_tanh_for_sigmoid`
- `use_jagged_space_indexing`
