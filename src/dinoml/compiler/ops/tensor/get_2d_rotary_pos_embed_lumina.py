from dinoml import backend
from dinoml.backend import registry
from dinoml.compiler.base import Operator, Tensor
from dinoml.compiler.dtype import normalize_dtype


class get_2d_rotary_pos_embed_lumina(Operator):
    def __init__(self):
        super().__init__()
        self._attrs["op"] = "get_2d_rotary_pos_embed_lumina"
        self._attrs["has_profiler"] = False
        self._attrs["nop"] = False

    def __call__(
        self,
        embed_dim: int,
        grid_size,
        linear_factor: float = 1.0,
        ntk_factor: float = 1.0,
        dtype: str = "float32",
    ):
        h, w = grid_size

        self._attrs["embed_dim"] = embed_dim
        self._attrs["grid_h"] = h
        self._attrs["grid_w"] = w
        self._attrs["linear_factor"] = float(linear_factor)
        self._attrs["ntk_factor"] = float(ntk_factor)
        self._attrs["dtype"] = normalize_dtype(dtype)
        self._attrs["inputs"] = []

        self._set_depth()

        y = Tensor(
            [h * w, embed_dim],
            src_ops={self},
            dtype=self._attrs["dtype"],
            skip_constant_folding=True,
        )
        self._attrs["outputs"] = [y]
        return y

    def gen_function(self):
        target = backend.target.Target.current()
        func_key = f"{target.name()}.{self._attrs['op']}.gen_function"
        func = registry.get(func_key)
        return func(self._attrs)
