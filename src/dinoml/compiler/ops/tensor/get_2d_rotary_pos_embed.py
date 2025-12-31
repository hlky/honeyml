from typing import Optional, Tuple

from dinoml import backend
from dinoml.backend import registry
from dinoml.compiler.base import IntImm, Operator, Tensor
from dinoml.compiler.dtype import normalize_dtype


class get_2d_rotary_pos_embed(Operator):
    """
    DinoML op version of get_2d_rotary_pos_embed (use_real=True only).

    Returns:
      cos: [grid_h*grid_w, embed_dim]
      sin: [grid_h*grid_w, embed_dim]
    """

    def __init__(self):
        super().__init__()
        self._attrs["op"] = "get_2d_rotary_pos_embed"
        self._attrs["has_profiler"] = False
        self._attrs["nop"] = False

    def __call__(
        self,
        embed_dim: int,
        crops_coords,
        grid_size,
        use_real: bool = True,
        theta: float = 10000.0,  # not in original signature but kept internal; default matches get_1d_rotary_pos_embed
        dtype: str = "float32",
    ):
        if use_real is not True:
            raise ValueError(
                "`use_real=False` is not supported in DinoML get_2d_rotary_pos_embed"
            )

        if embed_dim % 4 != 0:
            # python asserts embed_dim % 4 == 0
            raise ValueError("`embed_dim` must be divisible by 4")

        (start, stop) = crops_coords
        crop_start_h = float(start[0])
        crop_start_w = float(start[1])
        crop_stop_h = float(stop[0])
        crop_stop_w = float(stop[1])

        grid_h = grid_size[0]
        grid_w = grid_size[1]

        self._attrs["inputs"] = []
        self._attrs["dtype"] = normalize_dtype(dtype)

        # store same argument names
        self._attrs["embed_dim"] = embed_dim
        self._attrs["crop_start_h"] = crop_start_h
        self._attrs["crop_start_w"] = crop_start_w
        self._attrs["crop_stop_h"] = crop_stop_h
        self._attrs["crop_stop_w"] = crop_stop_w
        self._attrs["grid_h"] = grid_h
        self._attrs["grid_w"] = grid_w
        self._attrs["theta"] = float(theta)

        self._set_depth()

        n = grid_h * grid_w

        cos = Tensor(
            [n, embed_dim],
            src_ops={self},
            dtype=self._attrs["dtype"],
            skip_constant_folding=True,
        )
        sin = Tensor(
            [n, embed_dim],
            src_ops={self},
            dtype=self._attrs["dtype"],
            skip_constant_folding=True,
        )
        self._attrs["outputs"] = [cos, sin]
        return cos, sin

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = f"{target.name()}.{self._attrs['op']}.gen_function"
        func = registry.get(func_key)
        return func(self._attrs)
