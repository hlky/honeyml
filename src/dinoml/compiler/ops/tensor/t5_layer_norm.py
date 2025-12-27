from typing import List, Union

from dinoml import backend
from dinoml.backend import registry
from dinoml.compiler.base import IntVar, Operator, Tensor
from dinoml.compiler.dtype import normalize_dtype


class t5_layer_norm(Operator):
    def __init__(
        self,
        hidden_states: Tensor,
        weight: Tensor,
        eps: float = 1e-6,
        dtype: str = None,
    ):
        super().__init__()
        self._attrs["op"] = "t5_layer_norm"
        self._attrs["has_profiler"] = False
        self._attrs["nop"] = False

        self._attrs["eps"] = float(eps)

        # inputs
        self._attrs["inputs"] = [hidden_states, weight]

        # dtype: default to input dtype unless explicitly provided
        in_dtype = hidden_states._attrs["dtype"]
        self._attrs["dtype"] = normalize_dtype(dtype) if dtype is not None else in_dtype

    def __call__(self) -> Tensor:
        self._set_depth()

        out_shape = self._infer_shape()
        output = Tensor(
            out_shape,
            src_ops={self},
            dtype=self._attrs["dtype"],
        )
        self._attrs["outputs"] = [output]
        return output

    def _infer_shape(self) -> List[IntVar]:
        x = self._attrs["inputs"][0]
        w = self._attrs["inputs"][1]
        return x.shape()

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = f"{target.name()}.{self._attrs['op']}.gen_function"
        func = registry.get(func_key)
        return func(self._attrs)

    def _get_op_attributes(self):
        return {"eps": self._attrs["eps"]}
