from typing import Optional, Tuple, Union

from dinoml import backend
from dinoml.backend import registry
from dinoml.compiler.base import IntImm, Operator, Tensor
from dinoml.compiler.dtype import normalize_dtype


class get_3d_rotary_pos_embed(Operator):
    """
    DinoML op version of get_3d_rotary_pos_embed (use_real=True only).

    Produces:
      cos: [temporal_size * grid_size_h * grid_size_w, embed_dim]
      sin: [temporal_size * grid_size_h * grid_size_w, embed_dim]
    """

    def __init__(self):
        super().__init__()
        self._attrs["op"] = "get_3d_rotary_pos_embed"
        self._attrs["has_profiler"] = False
        self._attrs["nop"] = False

    def __call__(
        self,
        embed_dim: int,
        crops_coords,
        grid_size,
        temporal_size: int,
        theta: int = 10000,
        use_real: bool = True,
        grid_type: str = "linspace",
        max_size: Optional[Tuple[int, int]] = None,
        dtype: str = "float32",
    ) -> Tuple[Tensor, Tensor]:
        if use_real is not True:
            raise ValueError(
                "`use_real = False` is not currently supported for get_3d_rotary_pos_embed"
            )

        # Need even dims for each axis (since 1D RoPE uses dim%2==0)
        # dim_t = D/4 even and dim_h = 3*(D/8) even => requires D divisible by 16.
        if embed_dim % 16 != 0:
            raise ValueError(
                "`embed_dim` must be divisible by 16 for 3D RoPE (to keep axis dims even)."
            )

        # Parse crops_coords: ((start_h,start_w), (stop_h,stop_w))
        (start, stop) = crops_coords
        crop_start_h = float(start[0])
        crop_start_w = float(start[1])
        crop_stop_h = float(stop[0])
        crop_stop_w = float(stop[1])

        # grid_size: (H, W)
        grid_size_h = grid_size[0]
        grid_size_w = grid_size[1]

        if grid_type not in ("linspace", "slice"):
            raise ValueError("Invalid value passed for `grid_type`.")

        grid_type_enum = 0 if grid_type == "linspace" else 1

        if grid_type == "slice":
            if max_size is None:
                raise ValueError("`max_size` must be provided when grid_type='slice'.")
            max_h, max_w = int(max_size[0]), int(max_size[1])
            if grid_size_h > max_h or grid_size_w > max_w:
                raise ValueError(
                    "grid_size must be <= max_size when grid_type='slice'."
                )
        else:
            max_h, max_w = 0, 0  # unused

        # attrs for backend
        self._attrs["inputs"] = []
        self._attrs["dtype"] = normalize_dtype(dtype)

        # Use the same argument names
        self._attrs["embed_dim"] = int(embed_dim)
        self._attrs["crop_start_h"] = crop_start_h
        self._attrs["crop_start_w"] = crop_start_w
        self._attrs["crop_stop_h"] = crop_stop_h
        self._attrs["crop_stop_w"] = crop_stop_w
        self._attrs["grid_size_h"] = grid_size_h
        self._attrs["grid_size_w"] = grid_size_w
        self._attrs["temporal_size"] = temporal_size
        self._attrs["theta"] = float(theta)
        self._attrs["grid_type"] = int(grid_type_enum)
        self._attrs["max_h"] = int(max_h)
        self._attrs["max_w"] = int(max_w)

        self._set_depth()

        n = temporal_size * grid_size_h * grid_size_w

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
