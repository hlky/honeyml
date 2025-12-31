from typing import Tuple, Union

from dinoml import backend
from dinoml.backend import registry
from dinoml.compiler.base import IntImm, Operator, Tensor
from dinoml.compiler.dtype import normalize_dtype


class get_3d_sincos_pos_embed_cogvideox(Operator):
    """
    CogVideoX fused positional embedding op:

      joint = zeros([1, max_text_seq_length + (T*H*W), D])
      joint[:, max_text_seq_length:, :].copy_( get_3d_sincos_pos_embed(...).flatten(0,1) )

    Inputs:
      - no tensor inputs
    Output:
      - Tensor [1, max_text_seq_length + temporal_size*spatial_w*spatial_h, embed_dim]
    """

    def __init__(self):
        super().__init__()
        self._attrs["op"] = "get_3d_sincos_pos_embed_cogvideox"
        self._attrs["has_profiler"] = False
        self._attrs["nop"] = False

    def __call__(
        self,
        embed_dim: int,
        spatial_size: Union[int, Tuple[int, int]],
        temporal_size: int,
        max_text_seq_length: int,
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

        self._attrs["inputs"] = []
        self._attrs["dtype"] = normalize_dtype(dtype)

        self._attrs["embed_dim"] = embed_dim
        self._attrs["spatial_w"] = spatial_w  # matches torch: spatial_size[0]
        self._attrs["spatial_h"] = spatial_h  # matches torch: spatial_size[1]
        self._attrs["temporal_size"] = temporal_size
        self._attrs["max_text_seq_length"] = int(max_text_seq_length)
        self._attrs["spatial_interpolation_scale"] = float(spatial_interpolation_scale)
        self._attrs["temporal_interpolation_scale"] = float(
            temporal_interpolation_scale
        )

        self._set_depth()

        # output shape: [1, max_text_seq_length + (T*H*W), D]
        num_patches = (
            self._attrs["temporal_size"]
            * self._attrs["spatial_w"]
            * self._attrs["spatial_h"]
        )
        total_len = self._attrs["max_text_seq_length"] + num_patches

        y = Tensor(
            [IntImm(1), IntImm(total_len), IntImm(self._attrs["embed_dim"])],
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
