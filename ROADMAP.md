# DinoML Roadmap

This roadmap is based on a codebase pass across `src/dinoml/`, `csrc/`, `builder/`, `scripts/`, `docs/`, and `tests/`. DinoML already has a substantial compiler and backend surface; the next work should focus on making that surface easier to validate, extend, and ship.

## Current State

- The core architecture is strong and coherent: `compiler/` builds and optimizes graphs, `backend/{cuda,rocm}/` generates target code, and `csrc/` provides the runtime and shared-library interface.
- Model support is already meaningful. The repository includes builders and mappings for Stable Diffusion family models, Flux, PixArt, T5, ESRGAN, and T2I-Adapter.
- The test suite is large, especially under `tests/unittest/ops/` and `tests/unittest/compiler/`, which suggests the operator and graph-transform layers are the project’s deepest investment.
- The surrounding product surface is less mature. `docs/MODELING.md` is empty, `builder/README.md` says “Subject to change,” there is no CI config checked in, and builder dependencies are broader than the base package metadata.
- The repository currently mixes multiple entrypoint styles: package builders in `src/dinoml/builder/`, legacy root scripts in `builder/`, and CLI-style build scripts in `scripts/`.

## Phase 1: Repository Usability and Repeatability

Priority: make the project easier to install, understand, and validate.

1. Fill the documentation gaps.
   Add a real `docs/MODELING.md` describing the end-to-end flow from annotated model definitions to `compile_model()` to the generated runtime module. Document the role of `tmp/`, generated debug artifacts, and the CUDA/ROCm target selection path.
2. Define supported environments explicitly.
   Add dev and builder extras for packages currently used outside `install_requires`, especially `torch`, `diffusers`, `transformers`, `pytest`, `ruff`, `requests`, and `huggingface_hub`. Publish a tested version matrix for Python, CUDA, ROCm, PyTorch, and diffusers/transformers.
3. Add basic automation.
   Introduce CI for lint plus at least one smoke test tier. Even if full GPU coverage is too expensive, the repo needs a stable “this branch is sane” signal.
4. Segment the test suite.
   Separate fast static checks, CPU-safe unit tests, and GPU integration tests. Today, most useful validation appears to assume a live CUDA or ROCm environment.

## Phase 2: Builder and API Stabilization

Priority: make model compilation a consistent product surface instead of a collection of scripts.

1. Consolidate entrypoints.
   Choose one primary interface for compilation and demote the others to examples or compatibility wrappers. The duplication between `builder/`, `src/dinoml/builder/`, and `scripts/*_build.py` will slow maintenance as model coverage grows.
2. Harden configuration and weight loading.
   `src/dinoml/builder/config.py` fetches Hugging Face configs directly at runtime. Add clearer error handling, offline/cache-aware behavior, and a documented auth story for private or gated repos.
3. Add builder-focused regression coverage.
   The compiler and ops layers are well-tested; builder flows are not. Add smoke tests for a few representative builders such as AutoencoderKL, UNet2DCondition, Flux, and T5.
4. Normalize artifact naming and outputs.
   Formalize naming, metadata, and output layout for compiled modules so downstream tooling can discover and reuse them predictably.

## Phase 3: Backend Parity and Correctness

Priority: turn broad operator coverage into dependable cross-backend behavior.

1. Triage high-impact TODOs and `NotImplementedError`s.
   Focus on gaps that affect user-facing model support first: attention processors, normalization variants, pooling edge cases, and incomplete ROCm implementations.
2. Reduce backend drift.
   The codebase contains several comments about unifying CUDA and ROCm paths. Converge shared logic where practical so new ops and fixes do not fork immediately.
3. Improve profiling and cache reliability.
   The profiler/cache path is central to performance, but it still contains versioning and cleanup FIXME notes. Make cache invalidation, schema/version control, and minimal-algorithm selection deterministic.
4. Expand dynamic-shape coverage.
   Dynamic shapes are a core differentiator in the README. Add more targeted tests and model-level checks that prove dynamic behavior remains correct across codegen, runtime, and memory planning.

## Phase 4: Release Readiness and Ecosystem

Priority: make DinoML easier to consume outside local development.

1. Finish the packaging story.
   Add optional extras, release notes, and a versioned support policy. The repo should distinguish library consumers from model-builder users.
2. Complete serialization and portability features.
   `src/dinoml/utils/serialization/` is a clear expansion point. Decide whether serialized programs are a first-class artifact and finish or trim that surface accordingly.
3. Publish benchmark baselines.
   Ship repeatable benchmark recipes for a small set of canonical workloads and hardware targets so performance regressions are visible.
4. Improve contributor onboarding.
   Keep `AGENTS.md`, contributor docs, and architecture notes aligned so external contributors can work on ops, builders, or runtime code without rediscovering the same structure.

## Success Criteria

- A clean install path exists for both “library only” and “model builder” users.
- Lint and smoke validation run automatically on every change.
- At least one supported builder per major model family has a repeatable regression test.
- CUDA and ROCm support are documented with an explicit support matrix instead of inferred from source.
- Core docs explain how a model moves from Python graph construction to generated runtime code.
