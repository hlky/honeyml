from typing import List, Union
from dinoml import backend
from dinoml.backend import registry
from dinoml.compiler.base import IntVar, Operator, Tensor, IntImm
from dinoml.compiler.dtype import normalize_dtype


class relative_attention_bias(Operator):
    def __init__(
        self,
        embedding: Tensor,  # [num_buckets, n_heads]
        query_length: Union[int, IntVar],
        key_length: Union[int, IntVar],
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128,
        dtype: str = None,
    ):
        super().__init__()
        self._attrs["op"] = "relative_attention_bias"
        self._attrs["has_profiler"] = False
        self._attrs["nop"] = False

        self._attrs["inputs"] = [embedding]

        self._attrs["Q"] = (
            query_length if isinstance(query_length, IntVar) else IntImm(query_length)
        )
        self._attrs["K"] = (
            key_length if isinstance(key_length, IntVar) else IntImm(key_length)
        )

        self._attrs["bidirectional"] = bool(bidirectional)
        self._attrs["num_buckets"] = int(num_buckets)
        self._attrs["max_distance"] = int(max_distance)

        in_dtype = embedding._attrs["dtype"]
        self._attrs["dtype"] = normalize_dtype(dtype) if dtype is not None else in_dtype

    def __call__(self) -> Tensor:
        self._set_depth()

        emb = self._attrs["inputs"][0]
        H = emb.shape()[1]  # n_heads

        out_shape = [IntImm(1), H, self._attrs["Q"], self._attrs["K"]]
        y = Tensor(
            out_shape,
            src_ops={self},
            skip_constant_folding=True,
            dtype=self._attrs["dtype"],
        )
        self._attrs["outputs"] = [y]
        return y

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = f"{target.name()}.{self._attrs['op']}.gen_function"
        return registry.get(func_key)(self._attrs)

    def _get_op_attributes(self):
        return {
            "bidirectional": self._attrs["bidirectional"],
            "num_buckets": self._attrs["num_buckets"],
            "max_distance": self._attrs["max_distance"],
        }
