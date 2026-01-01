## Disable profilers

When you need fast iteration or using unsupported platform.

```python
import os
os.environ["DISABLE_PROFILER_CODEGEN"] = "1"
os.environ["CI_FLAG"] = "CIRCLECI"
```

## Lint

```sh
ruff format src scripts tests inference builder setup.py --exclude src/dinoml/utils/cutlass_lib/ --exclude src/dinoml/utils/ck_lib/ --exclude scripts/mk_cutlass_lib/ --exclude scripts/mk_ck_lib/
```

```sh
ruff check src scripts tests inference builder setup.py --exclude src/dinoml/utils/cutlass_lib/ src/dinoml/utils/ck_lib/ scripts/mk_cutlass_lib/ scripts/mk_ck_lib/ --ignore F821 --ignore F401 --ignore F841 --ignore F403 --ignore E741 --ignore E731 --ignore E402
```

## VSCode Intellisense

It is recommended to set `defines` for either `DINOML_CUDA` or `DINOML_HIP`, and add CUDA toolkit or HIP to includes.

On Windows `.vscode/c_cpp_properties.json` looks something like:

```json
{
    "configurations": [
        {
            "name": "Win32",
            "includePath": [
                "${workspaceFolder}/**",
                "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include\\**"
            ],
            "defines": [
                "_DEBUG",
                "UNICODE",
                "_UNICODE",
                "DINOML_CUDA"
            ],
            "windowsSdkVersion": "10.0.22621.0",
            "compilerPath": "cl.exe",
            "cStandard": "c17",
            "intelliSenseMode": "windows-msvc-x64"
        }
    ],
    "version": 4
}
```
