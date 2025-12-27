from typing import List, Union

from dinoml import backend
from dinoml.backend import registry
from dinoml.compiler.base import IntVar, Operator, Tensor
from dinoml.compiler.dtype import normalize_dtype


class gelu_new(Operator):
    def __init__(self):
        super().__init__()
        self._attrs["op"] = "gelu_new"
        self._attrs["has_profiler"] = False
        self._attrs["nop"] = False

    def __call__(self, x: Tensor) -> Tensor:
        self._attrs["inputs"] = [x]
        self._attrs["dtype"] = x._attrs["dtype"]

        self._set_depth()
        y = Tensor(
            self._attrs["inputs"][0].shape(),
            src_ops={self},
            dtype=self._attrs["dtype"],
        )
        self._attrs["outputs"] = [y]
        return y

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = f"{target.name()}.{self._attrs['op']}.gen_function"
        func = registry.get(func_key)
        return func(self._attrs)
