from dinoml import backend
from dinoml.backend import registry
from dinoml.compiler.base import Operator, Tensor


class fir_downsample2d(Operator):
    """
    FIR downsample 2D (fixed kernel (1,3,3,1), factor=2), NHWC -> NHWC.
    """

    def __init__(self):
        super().__init__()
        self._attrs["op"] = "fir_downsample2d"
        self._attrs["has_profiler"] = False
        self._attrs["nop"] = False

    def __call__(self, x: Tensor) -> Tensor:
        self._attrs["inputs"] = [x]
        self._attrs["dtype"] = x._attrs["dtype"]
        self._set_depth()

        # NHWC shape
        N, H, W, C = x.shape()
        y = Tensor([N, H / 2, W / 2, C], src_ops={self}, dtype=self._attrs["dtype"])
        self._attrs["outputs"] = [y]
        return y

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = f"{target.name()}.{self._attrs['op']}.gen_function"
        return registry.get(func_key)(self._attrs)
