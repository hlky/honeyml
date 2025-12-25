import copy
from typing import List

from honey import backend
from honey.backend import registry
from honey.compiler.base import IntVar, Operator, Tensor


class repeat_interleave(Operator):
    def __init__(self, repeats: int, repeat_dim: int):
        super().__init__()
        self._attrs["op"] = "repeat_interleave"
        self._attrs["has_profiler"] = False
        self._attrs["nop"] = False
        self._attrs["repeats"] = repeats
        self._attrs["repeat_dim"] = repeat_dim

    def __call__(self, x: Tensor) -> Tensor:
        self._attrs["inputs"] = [x]
        self._set_depth()
        output_shape = self._infer_shape(x)
        output = Tensor(output_shape, src_ops={self})
        self._attrs["outputs"] = [output]
        return output

    def _infer_shape(self, x) -> List[IntVar]:
        shape = copy.deepcopy(x.shape())
        shape[self._attrs["repeat_dim"]] *= self._attrs["repeats"]
        return shape

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = f"{target.name()}.{self._attrs['op']}.gen_function"
        func = registry.get(func_key)
        return func(self._attrs)

    def _get_op_attributes(self):
        return {
            "repeats": self._attrs["repeats"],
            "repeat_dim": self._attrs["repeat_dim"],
        }
