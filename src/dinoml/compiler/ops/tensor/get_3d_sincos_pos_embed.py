from typing import Tuple, Union

from dinoml import backend
from dinoml.backend import registry
from dinoml.compiler.base import IntImm, Operator, Tensor
from dinoml.compiler.dtype import normalize_dtype


class get_3d_sincos_pos_embed(Operator):
    """
    DinoML op version of:

      get_3d_sincos_pos_embed(
          embed_dim,
          spatial_size,
          temporal_size,
          spatial_interpolation_scale=1.0,
          temporal_interpolation_scale=1.0
      ) -> Tensor [T, H*W, D]

    Notes:
      - This op has no tensor inputs; it produces a positional embedding tensor.
      - Output dtype defaults to float32 (matching torch reference behavior).
    """

    def __init__(self):
        super().__init__()
        self._attrs["op"] = "get_3d_sincos_pos_embed"
        self._attrs["has_profiler"] = False
        self._attrs["nop"] = False

    def __call__(
        self,
        embed_dim: int,
        spatial_size: Union[int, Tuple[int, int]],
        temporal_size: int,
        spatial_interpolation_scale: float = 1.0,
        temporal_interpolation_scale: float = 1.0,
        dtype: str = "float32",
    ) -> Tensor:
        if embed_dim % 4 != 0:
            raise ValueError("`embed_dim` must be divisible by 4")

        if isinstance(spatial_size, int):
            spatial_w = spatial_size
            spatial_h = spatial_size
        else:
            spatial_w, spatial_h = spatial_size

        # store attrs used by backend codegen
        self._attrs["inputs"] = []
        self._attrs["dtype"] = normalize_dtype(dtype)

        self._attrs["embed_dim"] = embed_dim
        self._attrs["spatial_w"] = spatial_w  # matches torch: spatial_size[0]
        self._attrs["spatial_h"] = spatial_h  # matches torch: spatial_size[1]
        self._attrs["temporal_size"] = temporal_size
        self._attrs["spatial_interpolation_scale"] = float(spatial_interpolation_scale)
        self._attrs["temporal_interpolation_scale"] = float(
            temporal_interpolation_scale
        )

        self._set_depth()

        # output shape: [T, H*W, D]
        t = temporal_size
        hw = spatial_w * spatial_h
        d = embed_dim

        y = Tensor(
            [t, hw, d],
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
