from typing import Union
from dinoml.backend import registry

from dinoml.backend.target import Target

from dinoml.compiler.base import Operator, Tensor, IntImm, IntVar


class kdownsample2d_weight(Operator):
    def __init__(self):
        super().__init__()
        self._attrs["op"] = "kdownsample2d_weight"
        self._attrs["has_profiler"] = False

    def __call__(self, channels: Union[int, IntVar, IntImm], dtype: str) -> Tensor:
        self._attrs["channels"] = channels
        self._set_depth()
        out = Tensor(
            [channels, channels, 4, 4],
            src_ops={self},
            dtype=dtype,
            skip_constant_folding=True,
        )
        self._attrs["outputs"] = [out]
        return out

    def gen_function(self) -> str:
        target = Target.current()
        func = registry.get(f"{target.name()}.{self._attrs['op']}.gen_function")
        return func(self._attrs)
