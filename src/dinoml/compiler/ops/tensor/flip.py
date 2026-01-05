from typing import List, Sequence, Union

from dinoml import backend
from dinoml.backend import registry
from dinoml.compiler.base import Operator, Tensor


def _normalize_dims(dims: Union[int, Sequence[int]], rank: int) -> List[int]:
    if isinstance(dims, int):
        dims_list = [dims]
    else:
        dims_list = list(dims)

    if len(dims_list) == 0:
        return []

    norm: List[int] = []
    for d in dims_list:
        if d < 0:
            d = d + rank
        if d < 0 or d >= rank:
            raise RuntimeError(f"flip dims out of range: got {dims_list}, rank={rank}")
        norm.append(int(d))

    norm = sorted(set(norm))
    return norm


class flip(Operator):
    def __init__(self, dims: Union[int, Sequence[int]]):
        super().__init__()
        self._attrs["op"] = "flip"
        self._attrs["has_profiler"] = False
        self._attrs["nop"] = False
        self._attrs["dims"] = dims

    def __call__(self, x: Tensor) -> Tensor:
        self._attrs["inputs"] = [x]
        self._attrs["dtype"] = x._attrs["dtype"]

        rank = len(x._attrs["shape"])
        flip_dims = _normalize_dims(self._attrs["dims"], rank)
        self._attrs["flip_dims"] = flip_dims

        self._set_depth()
        y = Tensor(
            x.shape(),
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
