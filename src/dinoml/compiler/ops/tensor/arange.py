from typing import List, Union

from dinoml import backend
from dinoml.backend import registry
from dinoml.compiler.base import IntVar, Operator, Tensor
from dinoml.compiler.dtype import normalize_dtype


class arange(Operator):
    def __init__(
        self,
        start: Union[int, IntVar],
        stop: Union[int, IntVar],
        step: Union[int, IntVar],
        dtype: str = "float16",
    ):
        super().__init__()
        self._attrs["op"] = "arange"
        self._attrs["has_profiler"] = False
        self._attrs["nop"] = False
        self._attrs["start"] = start if isinstance(start, IntVar) else IntVar([start])
        self._attrs["stop"] = stop if isinstance(stop, IntVar) else IntVar([stop])
        self._attrs["step"] = step if isinstance(step, IntVar) else IntVar([step])
        self._attrs["dtype"] = normalize_dtype(dtype)

    def __call__(self) -> Tensor:
        self._attrs["inputs"] = []
        self._set_depth()
        output_shape = self._infer_shape()
        output = Tensor(
            output_shape,
            src_ops={self},
            skip_constant_folding=True,
            dtype=self._attrs["dtype"],
        )
        self._attrs["outputs"] = [output]
        return output

    def _infer_shape(self) -> List[IntVar]:
        num_elements = (self._attrs["stop"] - self._attrs["start"]) / self._attrs[
            "step"
        ]
        return [num_elements]

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = f"{target.name()}.{self._attrs['op']}.gen_function"
        func = registry.get(func_key)
        return func(self._attrs)

    def _get_op_attributes(self):
        return {
            "start": self._attrs["start"],
            "stop": self._attrs["stop"],
            "step": self._attrs["step"],
        }
