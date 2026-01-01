from dinoml import backend
from dinoml.compiler.base import Operator, Tensor
from dinoml.backend import registry


class sinusoidal_positional_embedding(Operator):
    def __init__(self):
        super().__init__()
        self._attrs["op"] = "sinusoidal_positional_embedding"
        self._attrs["has_profiler"] = False
        self._attrs["nop"] = False

    def __call__(self, x: Tensor, embed_dim: int, max_seq_len: int):
        self._attrs["inputs"] = [x]
        self._set_depth()
        self._attrs["dtype"] = x._attrs["dtype"]

        batch, seq, _ = x._attrs["shape"]
        self._attrs["embed_dim"] = embed_dim
        self._attrs["max_seq_len"] = max_seq_len

        out = Tensor(
            [batch, seq, embed_dim],
            src_ops={self},
            skip_constant_folding=True,
            dtype=self._attrs["dtype"],
        )
        self._attrs["outputs"] = [out]
        return out

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = f"{target.name()}.{self._attrs['op']}.gen_function"
        func = registry.get(func_key)
        return func(self._attrs)
