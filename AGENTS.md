# Repository Guidelines

## Project Structure & Module Organization

`src/dinoml/` is the main package. Core areas include `compiler/` for graph lowering and transforms, `backend/cuda/` and `backend/rocm/` for target-specific codegen, `frontend/` for public APIs, `modeling/` for model wrappers, and `utils/` for shared helpers. `csrc/` contains the C++ runtime and headers. Top-level `builder/` modules and `scripts/*_build.py` are model build entry points; `inference/` contains smoke-style runtime examples. `tests/unittest/` is the main regression suite, while `tests/modeling/` covers layer/model parity. `3rdparty/` is managed with git submodules, and `tmp/` is scratch output for local builds/tests.

## Build, Test, and Development Commands

- `git submodule update --init --recursive` syncs CUTLASS, CK, FlashAttention, and other vendored dependencies.
- `pip install -e .` installs DinoML in editable mode.
- `pip install diffusers transformers==4.57.3 pytest ruff` installs common modeling and dev tools not included in `install_requires`.
- `ruff format src scripts tests inference builder setup.py --exclude src/dinoml/utils/cutlass_lib --exclude src/dinoml/utils/ck_lib --exclude scripts/mk_cutlass_lib --exclude scripts/mk_ck_lib` formats first-party code.
- `ruff check src scripts tests inference builder setup.py --exclude src/dinoml/utils/cutlass_lib --exclude src/dinoml/utils/ck_lib --exclude scripts/mk_cutlass_lib --exclude scripts/mk_ck_lib --ignore F821,F401,F841,F403,E741,E731,E402` runs the repo’s current lint policy.
- `python -m unittest discover -s tests/unittest -p 'test_*.py'` runs the main unit suite.
- `python -m unittest discover -s tests/modeling -p '*.py'` runs modeling tests with nonstandard filenames.
- `python -m pytest tests/unittest/util/test_debug_utils.py` runs the pytest-only debug utility checks.

## Coding Style & Naming Conventions

Use 4-space indentation, `snake_case` for modules/functions, and `PascalCase` for classes, including `*TestCase` test classes. Keep imports grouped as standard library, third-party, then local. Follow existing type-hint patterns on new public helpers. Avoid hand-editing generated kernel trees under `src/dinoml/utils/{cutlass_lib,ck_lib}` and `scripts/mk_{cutlass,ck}_lib/` unless you are intentionally regenerating them.

## Testing Guidelines

Many tests compile and run GPU kernels, so validate on the intended CUDA or ROCm target and call out backend assumptions in reviews. No coverage threshold is configured; add focused regression tests for compiler passes, backend codegen, and model builders you change.

## Commit & Pull Request Guidelines

Recent commits use short, capitalized subjects with optional PR suffixes, for example `Dynamic workspace (#20)` and `Updates for AMD/ROCm (#37)`. Keep commits narrow in scope. PRs should summarize backend or model impact, list the commands you ran, note relevant environment flags such as `DISABLE_PROFILER_CODEGEN=1` or `CI_FLAG=CIRCLECI`, and avoid committing `tmp/` artifacts or accidental submodule pointer changes.
