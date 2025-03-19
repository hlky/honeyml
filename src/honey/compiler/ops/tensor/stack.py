from typing import List

from honey.compiler.base import Operator, Tensor
from honey.compiler.ops.common.view_ops import unsqueeze
from honey.compiler.ops.tensor.concatenate import concatenate


class stack(Operator):
    def __init__(self) -> None:
        super().__init__()
        self._attrs["op"] = "stack"

    def __call__(self, inputs: List[Tensor], dim=0) -> List[Tensor]:
        return concatenate()([unsqueeze(dim)(tensor) for tensor in inputs], dim=dim)
