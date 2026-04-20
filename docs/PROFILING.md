# Profiling

DinoML uses the word "profiling" for two different workflows:

1. compile-time kernel profiling
   Used during `compile_model(...)` to pick the fastest kernel implementation for an op workload.
2. runtime module profiling
   Used after compilation to measure an already-built module and write a timing report.

This document covers both, with emphasis on the compile-time path because it directly affects generated code.

## Compile-Time Profiling

The compile-time profiling pass lives in `src/dinoml/compiler/transform/profile.py`.

At a high level, DinoML:

1. walks the graph and asks each profiled op to emit profiler sources with `backend.codegen.gen_profiler(...)`
2. compiles those profiler sources into standalone executables under `tmp/<model_name>/profiler/`
3. runs the profiler executables on one or more GPUs
4. records the winning algorithm into each op's `exec_path`
5. stores results in the local SQLite profiling cache when supported

Ops with `has_profiler=False` skip this entirely. Profiling is mainly relevant for GEMM, conv, conv3d, normalization, softmax, topk, argmax, and selected vision ops.

## Profiling Cache

The local cache is managed by `src/dinoml/backend/profiler_cache.py` and loaded through the active target in `src/dinoml/backend/target.py`.

By default the cache lives at:

- `~/.dinoml/cuda.db`
- `~/.dinoml/rocm.db`

The cache key is not just the op type. It also includes workload identity such as shape signature hash, layouts, dtypes, epilogue, and target architecture.

Important details:

- GEMM, conv2d, and conv3d caches are versioned.
- normalization uses the same cache DB but does not currently expose a separate version property.
- old cache tables are intentionally kept when versions change.
- a cache hit can skip profiler generation for static workloads.

## Profiling-Related Environment Variables

The profiling path is controlled by a mix of DinoML-specific and legacy environment variables:

- `DINOML_FORCE_PROFILER_CACHE=1`
  Require cache hits instead of launching profilers. Compilation fails on a cache miss.
- `FORCE_PROFILE=1`
  Force live profiling even in CI-like environments and even when profiler codegen would otherwise be disabled.
- `DISABLE_PROFILER_CODEGEN=1`
  Skip profiler code generation unless `FORCE_PROFILE=1` is also set.
- `CACHE_DIR=/path/to/cache`
  Override the directory used for the local profiling cache database.
- `FLUSH_PROFILE_CACHE=1`
  Remove the target cache DB before loading it.
- `CI_FLAG=CIRCLECI`
  Marks the run as CI for target logic.
- `TRICK_CI_ENV=1`
  Overrides CI detection in places where you need normal profiling behavior even if `CI_FLAG` is set.

## Dynamic Shapes

Dynamic-shape profiling is more constrained than static profiling.

Current behavior in the codebase:

- GEMM, softmax, layernorm, and groupnorm use `DynamicProfileStrategy` to reduce dynamic ranges to representative profiling shapes.
- conv2d and conv3d only support `DynamicProfileStrategy.HINTS` for dynamic profiling.
- group GEMM has explicit FIXME notes around full `dynamic_profiling_strategy` support.

Two important caveats for dynamic conv profiling:

- DinoML may still need profiler binaries even when static cache entries exist, because dynamic boundary refinement uses those binaries after the initial static decisions.
- `DINOML_FORCE_PROFILER_CACHE=1` is incompatible with dynamic conv/conv3d profiling and raises at compile time.

This behavior is covered by tests such as `tests/unittest/ops/test_conv_profiler_cache.py` and `tests/unittest/ops/test_conv3d_profiler_cache.py`.

## Compile-Time Runtime Model

Compile-time profilers are not the same as the runtime `Model.profile(...)` API.

Compile-time profiling happens before the final shared library is produced and selects kernel implementations.
Runtime profiling happens after compilation and measures the already-selected implementation.

## Runtime Module Profiling

The Python wrapper exposes:

- `module.profile(inputs, outputs, num_iters, filename)`
- `module.profile_with_tensors(inputs, outputs, num_iters, filename)`
- `module.benchmark(...)`
- `module.benchmark_with_tensors(...)`

These APIs live in `src/dinoml/compiler/model.py` and call into the C++ runtime through `DinoMLModelContainerProfile` and `DinoMLModelContainerBenchmark`.

Use them when you want to answer:

- how expensive is this compiled module on a real input shape?
- which kernels dominate runtime?
- how does graph mode or multithreading affect latency?

Use compile-time profiling when you want to answer:

- which kernel variant should DinoML bake into the generated execution path?
- is the cache warm or missing entries?
- did an op start generating new workloads after a code change?

## Multi-GPU Profiling

Compile-time profilers can run across multiple devices. `ProfilerRunner` coordinates subprocess execution and binds each task to a device through the target-specific visibility variable such as `CUDA_VISIBLE_DEVICES` or `HIP_VISIBLE_DEVICES`.

GEMM uses a separate `ProfilerRunner` implementation that supports asynchronous result collection and postprocessing across split-k candidates.

## Artifacts and Where To Look

Compile-time profiling artifacts are usually written below:

- `tmp/<model_name>/profiler/`

Useful places to inspect:

- generated profiler sources and executables in `tmp/<model_name>/profiler/`
- compile logs from `dinoml.compiler.transform.profile`
- cache load/version logs from `dinoml.backend.profiler_cache`
- op-specific profile logic in:
  `src/dinoml/compiler/ops/gemm_universal/gemm_common.py`
  `src/dinoml/compiler/ops/conv/conv2d.py`
  `src/dinoml/compiler/ops/conv/conv3d.py`
  `src/dinoml/compiler/ops/layernorm/layernorm.py`
  `src/dinoml/compiler/ops/softmax/softmax.py`

## Practical Recommendations

- For normal local development, let DinoML use the local cache and only flush it when profiling-related code changes.
- Use `DINOML_FORCE_PROFILER_CACHE=1` when you want to prove a change does not introduce new profiling workloads, but avoid it for dynamic conv/conv3d cases.
- Use `FORCE_PROFILE=1` when debugging CI-only behavior or when `DISABLE_PROFILER_CODEGEN=1` would otherwise suppress profiling.
- Treat `module.profile(...)` and `module.benchmark(...)` as post-compilation measurement tools, not kernel-selection tools.

## Related Files

- `src/dinoml/compiler/transform/profile.py`
- `src/dinoml/backend/profiler_runner.py`
- `src/dinoml/backend/profiler_cache.py`
- `src/dinoml/backend/target.py`
- `src/dinoml/compiler/model.py`
- `tests/unittest/backend/test_profiler.py`
- `tests/unittest/ops/test_gemm_profiler_cache.py`
- `tests/unittest/ops/test_conv_profiler_cache.py`
- `tests/unittest/ops/test_conv3d_profiler_cache.py`
