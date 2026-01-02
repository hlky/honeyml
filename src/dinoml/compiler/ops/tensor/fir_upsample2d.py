from dinoml import backend
from dinoml.backend import registry
from dinoml.compiler.base import Operator, Tensor


class fir_upsample2d(Operator):
    def __init__(self):
        super().__init__()
        self._attrs["op"] = "fir_upsample2d"
        self._attrs["has_profiler"] = False

    def __call__(self, x: Tensor):
        self._attrs["inputs"] = [x]
        self._set_depth()
        self._attrs["dtype"] = x._attrs["dtype"]
        N, H, W, C = x.shape()
        y = Tensor([N, H * 2, W * 2, C], src_ops={self}, dtype=x._attrs["dtype"])
        self._attrs["outputs"] = [y]
        return y

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = f"{target.name()}.{self._attrs['op']}.gen_function"
        return registry.get(func_key)(self._attrs)
