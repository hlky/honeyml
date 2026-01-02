from typing import List

from dinoml.compiler.base import Tensor
from dinoml.compiler.ops.upsample.upsampling3d_base import upsampling3d_base


class upsampling3d_add(upsampling3d_base):
    def __init__(self, scale_factor, mode, align_corners=False) -> None:
        super().__init__(scale_factor, mode, align_corners)
        self._attrs["op"] = "upsampling3d_add"
        self._attrs["mode"] = mode

    def __call__(self, x: Tensor, r: Tensor) -> List[Tensor]:
        self._attrs["inputs"] = [x, r]
        self._set_depth()
        self._extract_exec_path(x)
        output_shape = self._infer_shapes(x)
        output = Tensor(output_shape, src_ops={self}, dtype=x._attrs["dtype"])
        self._attrs["outputs"] = [output]
        return output
