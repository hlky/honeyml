## Disable profilers

When you need fast iteration or using unsupported platform.

```python
import os
os.environ["DISABLE_PROFILER_CODEGEN"] = "1"
os.environ["CI_FLAG"] = "CIRCLECI"
```

## Lint

```sh
ruff format src scripts tests inference builder setup.py --exclude src/honey/utils/cutlass_lib/ --exclude src/honey/utils/ck_lib/ --exclude scripts/mk_cutlass_lib/ --exclude scripts/mk_ck_lib/
```

```sh
ruff check src scripts tests inference builder setup.py --exclude src/honey/utils/cutlass_lib/ src/honey/utils/ck_lib/ scripts/mk_cutlass_lib/ scripts/mk_ck_lib/ --ignore F821 --ignore F401 --ignore F841 --ignore F403 --ignore E741 --ignore E731 --ignore E402
```
