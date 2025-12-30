from typing import Union

from dinoml import backend
from dinoml.backend import registry
from dinoml.compiler.base import IntImm, Operator, Tensor
from dinoml.compiler.dtype import normalize_dtype


class get_timestep_embedding(Operator):
    def __init__(self):
        super().__init__()
        self._attrs["op"] = "get_timestep_embedding"
        self._attrs["has_profiler"] = False
        self._attrs["nop"] = False

    def __call__(
        self,
        timesteps: Tensor,
        embedding_dim: int,
        flip_sin_to_cos: bool = False,
        downscale_freq_shift: float = 1.0,
        scale: float = 1.0,
        max_period: int = 10000,
    ) -> Tensor:
        # Store inputs
        self._attrs["inputs"] = [timesteps]

        # Store op attributes (use the same argument names as requested)
        self._attrs["embedding_dim"] = int(embedding_dim)
        self._attrs["flip_sin_to_cos"] = bool(flip_sin_to_cos)
        self._attrs["downscale_freq_shift"] = float(downscale_freq_shift)
        self._attrs["scale"] = float(scale)
        self._attrs["max_period"] = int(max_period)

        # Output is float32 (matches the reference implementation behavior)
        self._attrs["dtype"] = normalize_dtype("float32")

        self._set_depth()

        # timesteps is 1-D: [N] -> output [N, embedding_dim]
        out_shape = [timesteps._attrs["shape"][0], IntImm(int(embedding_dim))]

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
        func = registry.get(func_key)
        return func(self._attrs)
