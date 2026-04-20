# Architecture

This document explains how DinoML moves from a symbolic Python model to a compiled shared library and runtime wrapper.

## Layers

DinoML is split into four main layers:

1. Frontend and modeling
   `src/dinoml/frontend/` provides `Tensor`, `IntVar`, and `nn.Module` building blocks. `src/dinoml/modeling/` contains higher-level model definitions, often mirroring Diffusers or Transformers architectures.
2. Compiler
   `src/dinoml/compiler/` owns graph construction, graph transforms, profiling, constant folding, memory planning, and the Python `Model` wrapper.
3. Backend code generation
   `src/dinoml/backend/` lowers graph nodes into target-specific source code for CUDA or ROCm and emits the final model driver sources.
4. Runtime
   `csrc/` contains the static C++ runtime implementation and the exported C ABI that the generated library exposes.

## End-to-End Compile Flow

`compile_model(...)` in `src/dinoml/compiler/compiler.py` is the main entry point after a symbolic graph has been created.

The compile pipeline is:

1. Toposort the output tensors into a graph.
2. Bind compile-time constants to the graph.
3. Remove unused ops and no-op nodes.
4. Name tensors and deduplicate symbolic names.
5. Mark parameter tensors and run graph optimization passes.
6. Mark special views and refine the graph.
7. Profile candidate kernels on the selected target.
8. Constant-fold subgraphs that can be precomputed.
9. Run memory planning for intermediates, constants, and workspace.
10. Generate per-op source files with `backend.codegen.gen_function_src(...)`.
11. Generate model-driver/runtime glue with `backend.codegen.gen_library_src(...)`.
12. Compile and link everything into a shared library through the backend build engine.
13. Return a Python `Model` object that wraps the compiled library.

Builders in `src/dinoml/builder/` automate the earlier modeling steps, but they still end at this same compile path.

## Build Directory Layout

Most generated artifacts are written under `tmp/<model_name>/`.

Typical contents include:

- graph debug dumps for passes such as `toposort`, `optimize_graph`, `profile`, `constant_folding`, and `memory_planning`
- generated per-op source files such as `<op_name><target extension>`
- generated model/runtime files including `model-generated.h`, `device_functions-generated.h`, and `model_container_base<target extension>`
- `constants.bin` for owned constants embedded into the compiled module
- object files and the final shared library, usually `<model_name>.so`

If compilation fails, `tmp/<model_name>/` is the first place to inspect.

## Generated Code Boundary

The split between static runtime code and generated code is deliberate:

- `csrc/` contains runtime code that is shared across all models.
- `backend/codegen.py` emits model-specific code for tensor setup, kernel launches, output shape handling, constant metadata, and runtime glue.

Two generated pieces matter most:

- `model-generated.h`
  This is the model-specific runtime implementation. It contains the code that sets up inputs/outputs/constants and launches the compiled kernels.
- `model_container_base{ext}`
  This is the generated portion of `ModelContainerBase`, including model-specific metadata such as parameter names, max shapes, constant offsets, and workspace sizing.

## Runtime Model

The exported ABI lives in `csrc/include/model_interface.h`. The Python wrapper in `src/dinoml/compiler/model.py` uses `ctypes` to call into that ABI.

At runtime there are two important C++ concepts:

- `Model`
  A generated runtime object derived from `ModelBase` in `csrc/include/model.h`. It owns blob/workspace pointers, sets up inputs and outputs, and runs the compiled kernel sequence.
- `ModelContainer`
  A higher-level object defined in `csrc/include/model_container.h`. It owns shared constants and a pool of `Model` runtimes so multiple inferences can run concurrently.

`compile_model(...)` returns the Python `Model` wrapper, but that wrapper is calling into a compiled `ModelContainer`.

## Constants

DinoML distinguishes between:

- bound constants
  These are known during compilation, can participate in constant folding, and are packaged into `constants.bin`
- unbound constants
  These are required by the runtime but are not known at compile time and must be supplied later with `set_constant(...)`

This distinction is part of the generated metadata in `model_container_base{ext}` and is enforced by the runtime.

## Dynamic Shapes and Outputs

Dynamic shapes are represented symbolically in the Python graph, then materialized in the generated runtime through shape pointers and bucket conditions.

One important runtime rule follows from this: outputs must be allocated at their maximum shape before `run()` or `run_with_tensors()` is called. The runtime then writes back the actual output shapes for that execution.

This is why the Python wrapper exposes methods such as:

- `get_input_maximum_shape(...)`
- `get_output_maximum_shape(...)`
- `get_output_name_to_index_map(...)`

## Concurrency, Streams, and Graph Mode

`ModelContainer` stores a pool of runtimes, controlled by `num_runtimes`. Each call to `run()` uses an available runtime instance and blocks only when all runtimes are busy.

The runtime also supports:

- explicit stream handles
- optional synchronization
- CUDA graph mode for repeated launches with high kernel-launch overhead

Graph mode is CUDA-only and is implemented inside the generated runtime through `ModelBase`.

## Target-Specific Pieces

The target object in `src/dinoml/backend/target.py` selects:

- source file extension and compiler toolchain
- CUDA vs ROCm templates and libraries
- profiler behavior
- build commands and link strategy

Builders usually call `dinoml.testing.detect_target(...)` to choose a target automatically from the active GPU environment.

## Where To Debug

Use these paths depending on the failure mode:

- modeling or input-shape issues
  `docs/MODELING.md`, `src/dinoml/utils/build_utils.py`, and the relevant builder in `src/dinoml/builder/`
- graph-transform or constant-folding issues
  `src/dinoml/compiler/` and the pass dumps in `tmp/<model_name>/`
- code generation issues
  `src/dinoml/backend/codegen.py`, `src/dinoml/backend/main_templates.py`, and the generated sources in `tmp/<model_name>/`
- runtime or invocation issues
  `src/dinoml/compiler/model.py`, `csrc/include/model_interface.h`, `csrc/include/model_container.h`, and `csrc/include/model.h`

## Related Docs

- `docs/MODELING.md` for symbolic model construction and builders
- `docs/MODULES.md` for module-building workflows
- `docs/PROFILING.md` for compile-time kernel selection and runtime measurement
- `docs/ENVIRONMENT.md` for compiler and runtime environment flags
- `csrc/README.md` for runtime-specific API notes
