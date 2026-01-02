from dinoml.compiler.base import Operator, Tensor
from dinoml.backend import registry


class prepare_for_transposed_conv2d(Operator):
    def __init__(self, stride):
        super().__init__()
        self._attrs["op"] = "prepare_for_transposed_conv2d"
        self._attrs["stride"] = tuple(stride)

    def __call__(self, x: Tensor) -> Tensor:
        self._attrs["inputs"] = [x]
        self._set_depth()
        self._attrs["dtype"] = x._attrs["dtype"]

        n, h, w, c = x.shape()
        sh, sw = self._attrs["stride"]

        out = Tensor(
            [n, (h - 1) * sh + 1, (w - 1) * sw + 1, c],
            dtype=x._attrs["dtype"],
            src_ops={self},
        )
        self._attrs["outputs"] = [out]
        return out

    def gen_function(self):
        return registry.get(f"{self._attrs['op']}.gen_function")(self._attrs)
