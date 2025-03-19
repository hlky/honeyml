from typing import List

from honey.compiler.base import Tensor
from honey.compiler.ops.upsample.upsampling1d_base import upsampling1d_base


# pylint: disable=C0103
class upsampling1d_add(upsampling1d_base):
    def __init__(self, scale_factor, mode, align_corners=False) -> None:
        super().__init__(scale_factor, mode, align_corners)
        self._attrs["op"] = "upsampling1d_add"
        self._attrs["mode"] = mode

    def __call__(self, x: Tensor, r: Tensor) -> List[Tensor]:
        self._attrs["inputs"] = [x, r]
        self._set_depth()
        self._extract_exec_path(x)
        output_shape = self._infer_shapes(x)
        output = Tensor(output_shape, src_ops={self}, dtype=x._attrs["dtype"])
        self._attrs["outputs"] = [output]
        return output
