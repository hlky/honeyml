import math
from dinoml import backend
from dinoml.backend import registry
from dinoml.compiler.base import Operator, Tensor
from dinoml.testing import detect_target


def workspace_size(q: Tensor) -> int:
    size = 4 * math.prod(q.shape()[i].upper_bound() for i in range(len(q.shape()) - 1))
    return (size + 16 - 1) // 16 * 16


class flash_attn(Operator):
    def __init__(
        self,
        mask_type: str = "none",
        window_size_left: int = -1,
        window_size_right: int = -1,
    ):
        super().__init__()
        self._attrs["op"] = "flash_attn"
        self._attrs["has_profiler"] = False
        self._attrs["nop"] = False

        self._attrs["mask_type"] = mask_type
        self._attrs["window_size_left"] = int(window_size_left)
        self._attrs["window_size_right"] = int(window_size_right)
        self._attrs["workspace"] = 0

    def __call__(self, q: Tensor, k: Tensor, v: Tensor):
        self._attrs["inputs"] = [q, k, v]
        self._attrs["dtype"] = q._attrs["dtype"]
        target = detect_target()
        if target.name() == "cuda":
            self._attrs["workspace"] = workspace_size(q)

        self._set_depth()

        out = Tensor(
            q.shape(),
            src_ops={self},
            dtype=self._attrs["dtype"],
        )

        self._attrs["outputs"] = [out]
        return out

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = f"{target.name()}.{self._attrs['op']}.gen_function"
        func = registry.get(func_key)
        return func(self._attrs)
