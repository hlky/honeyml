from typing import List, Optional, Union
from dinoml import backend
from dinoml.backend import registry
from dinoml.compiler.base import IntImm, IntVar, Operator, Tensor
from dinoml.compiler.dtype import normalize_dtype


class randn(Operator):
    def __init__(
        self,
        shape: List[Union[int, IntVar]],
        dtype: str,
        mean: float = 0.0,
        std: float = 1.0,
        seed: Optional[int] = None,
        offset_groups: Optional[int] = None,
    ):
        super().__init__()
        self._attrs["op"] = "randn"
        self._attrs["has_profiler"] = False
        self._attrs["nop"] = False
        self._attrs["inputs"] = []

        self._attrs["shape"] = [
            s if isinstance(s, IntVar) else IntImm(s) for s in shape
        ]
        self._attrs["mean"] = float(mean)
        self._attrs["std"] = float(std)
        self._attrs["seed"] = seed
        if seed is not None and offset_groups is None:
            offset_groups = 0
        self._attrs["offset_groups"] = offset_groups
        self._attrs["dtype"] = normalize_dtype(dtype)

    def __call__(self) -> Tensor:
        self._set_depth()
        y = Tensor(
            self._attrs["shape"],
            src_ops={self},
            dtype=self._attrs["dtype"],
            skip_constant_folding=True,
        )
        self._attrs["outputs"] = [y]
        return y

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        template_path = target.template_path()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(
            self._attrs,
            template_path,
        )

    def _get_op_attributes(self):
        return {
            "mean": self._attrs["mean"],
            "std": self._attrs["std"],
            "seed": self._attrs["seed"],
            "offset_groups": self._attrs["offset_groups"],
        }
