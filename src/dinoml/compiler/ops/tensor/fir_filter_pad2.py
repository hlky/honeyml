from dinoml import backend
from dinoml.backend import registry
from dinoml.compiler.base import Operator, Tensor


class fir_filter_pad2(Operator):
    """
    FIR filter stage for diffusers FirDownsample2D(use_conv=True):
    NHWC [N,H,W,C] -> NHWC [N,H+1,W+1,C]
    Uses fixed kernel (1,3,3,1) normalized, pad=2, stride=1, depthwise.
    """

    def __init__(self):
        super().__init__()
        self._attrs["op"] = "fir_filter_pad2"
        self._attrs["has_profiler"] = False
        self._attrs["nop"] = False

    def __call__(self, x: Tensor) -> Tensor:
        self._attrs["inputs"] = [x]
        self._attrs["dtype"] = x._attrs["dtype"]
        self._set_depth()

        N, H, W, C = x.shape()
        y = Tensor([N, H + 1, W + 1, C], src_ops={self}, dtype=self._attrs["dtype"])
        self._attrs["outputs"] = [y]
        return y

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = f"{target.name()}.{self._attrs['op']}.gen_function"
        return registry.get(func_key)(self._attrs)
