## Disable profilers

When you need fast iteration or using unsupported platform.

```python
import os
os.environ["DISABLE_PROFILER_CODEGEN"] = "1"
os.environ["CI_FLAG"] = "CIRCLECI"
```
