from typing import Tuple, Union

from dinoml import backend
from dinoml.backend import registry
from dinoml.compiler.base import Operator, Tensor
from dinoml.compiler.dtype import normalize_dtype


class get_2d_sincos_pos_embed(Operator):
    def __init__(self):
        super().__init__()
        self._attrs["op"] = "get_2d_sincos_pos_embed"
        self._attrs["has_profiler"] = False
        self._attrs["nop"] = False

    def __call__(
        self,
        embed_dim: int,
        grid_size: Union[int, Tuple[int, int]],
        cls_token: bool = False,
        extra_tokens: int = 0,
        interpolation_scale: float = 1.0,
        base_size: int = 16,
        dtype: str = "float32",
    ) -> Tensor:
        if embed_dim % 2 != 0:
            raise ValueError("embed_dim must be divisible by 2")

        if isinstance(grid_size, int):
            grid_h = grid_size
            grid_w = grid_size
        else:
            grid_h, grid_w = grid_size

        # attrs for backend
        self._attrs["inputs"] = []
        self._attrs["dtype"] = normalize_dtype(dtype)

        self._attrs["embed_dim"] = embed_dim
        self._attrs["grid_h"] = grid_h  # matches reference grid_size[0]
        self._attrs["grid_w"] = grid_w  # matches reference grid_size[1]
        self._attrs["cls_token"] = bool(cls_token)
        self._attrs["extra_tokens"] = int(extra_tokens)
        self._attrs["interpolation_scale"] = float(interpolation_scale)
        self._attrs["base_size"] = int(base_size)

        self._set_depth()

        # output shape: [rows, embed_dim]
        prepend = extra_tokens if (cls_token and extra_tokens > 0) else 0
        if prepend > 0:
            rows = prepend + (grid_h * grid_w)
        else:
            rows = grid_h * grid_w

        y = Tensor(
            [rows, embed_dim],
            src_ops={self},
            dtype=self._attrs["dtype"],
            skip_constant_folding=True,
        )
        self._attrs["outputs"] = [y]
        return y

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = f"{target.name()}.{self._attrs['op']}.gen_function"
        func = registry.get(func_key)
        return func(self._attrs)
