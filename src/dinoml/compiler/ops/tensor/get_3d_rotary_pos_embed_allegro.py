from typing import Tuple

from dinoml import backend
from dinoml.backend import registry
from dinoml.compiler.base import IntImm, Operator, Tensor
from dinoml.compiler.dtype import normalize_dtype


class get_3d_rotary_pos_embed_allegro(Operator):
    """
    DinoML op version of get_3d_rotary_pos_embed_allegro.

    Returns:
      (freqs_t, freqs_h, freqs_w), (grid_t, grid_h, grid_w)
    where:
      freqs_* is (cos, sin) each float32 with shapes:
        t: [T, D/3]
        h: [H, D/3]
        w: [W, D/3]
      grid_* are int64 with shape:
        [1, T*H*W]
      and grid ordering matches torch.cartesian_prod(grid_t, grid_h, grid_w)
    """

    def __init__(self):
        super().__init__()
        self._attrs["op"] = "get_3d_rotary_pos_embed_allegro"
        self._attrs["has_profiler"] = False
        self._attrs["nop"] = False

    def __call__(
        self,
        height: int,
        width: int,
        num_frames: int,
        vae_scale_factor_spatial: int = 8,
        patch_size: int = 2,
        interpolation_scale_h: float = 2.0,
        interpolation_scale_t: float = 2.2,
        interpolation_scale_w: float = 2.0,
        attention_head_dim: int = 96,
        dtype: str = "float32",
    ):
        if attention_head_dim % 3 != 0:
            raise ValueError(
                "`attention_head_dim` must be divisible by 3 for allegro RoPE."
            )
        if (attention_head_dim // 3) % 2 != 0:
            raise ValueError(
                "`attention_head_dim//3` must be divisible by 2 (1D RoPE dim must be even)."
            )

        self._attrs["inputs"] = []
        self._attrs["dtype"] = normalize_dtype(dtype)

        self._attrs["height"] = height
        self._attrs["width"] = width
        self._attrs["num_frames"] = num_frames
        self._attrs["vae_scale_factor_spatial"] = int(vae_scale_factor_spatial)
        self._attrs["patch_size"] = int(patch_size)
        self._attrs["interpolation_scale_h"] = float(interpolation_scale_h)
        self._attrs["interpolation_scale_t"] = float(interpolation_scale_t)
        self._attrs["interpolation_scale_w"] = float(interpolation_scale_w)
        self._attrs["attention_head_dim"] = int(attention_head_dim)

        self._set_depth()

        if isinstance(height, int):
            grid_h = height // (vae_scale_factor_spatial * patch_size)
        else:
            grid_h = height / (vae_scale_factor_spatial * patch_size)
        if isinstance(width, int):
            grid_w = width // (vae_scale_factor_spatial * patch_size)
        else:
            grid_w = width / (vae_scale_factor_spatial * patch_size)
        T = num_frames
        H = grid_h
        W = grid_w

        dim_axis = int(attention_head_dim) // 3
        N = T * H * W

        # freqs outputs (float)
        t_cos = Tensor(
            [T, dim_axis],
            src_ops={self},
            dtype=self._attrs["dtype"],
            skip_constant_folding=True,
        )
        t_sin = Tensor(
            [T, dim_axis],
            src_ops={self},
            dtype=self._attrs["dtype"],
            skip_constant_folding=True,
        )
        h_cos = Tensor(
            [H, dim_axis],
            src_ops={self},
            dtype=self._attrs["dtype"],
            skip_constant_folding=True,
        )
        h_sin = Tensor(
            [H, dim_axis],
            src_ops={self},
            dtype=self._attrs["dtype"],
            skip_constant_folding=True,
        )
        w_cos = Tensor(
            [W, dim_axis],
            src_ops={self},
            dtype=self._attrs["dtype"],
            skip_constant_folding=True,
        )
        w_sin = Tensor(
            [W, dim_axis],
            src_ops={self},
            dtype=self._attrs["dtype"],
            skip_constant_folding=True,
        )

        # grids (int64) shaped [1, N]
        grid_t = Tensor(
            [1, N],
            src_ops={self},
            dtype="int64",
            skip_constant_folding=True,
        )
        grid_h_t = Tensor(
            [1, N],
            src_ops={self},
            dtype="int64",
            skip_constant_folding=True,
        )
        grid_w_t = Tensor(
            [1, N],
            src_ops={self},
            dtype="int64",
            skip_constant_folding=True,
        )

        self._attrs["outputs"] = [
            t_cos,
            t_sin,
            h_cos,
            h_sin,
            w_cos,
            w_sin,
            grid_t,
            grid_h_t,
            grid_w_t,
        ]
        return (
            ((t_cos, t_sin), (h_cos, h_sin), (w_cos, w_sin)),
            (grid_t, grid_h_t, grid_w_t),
        )

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = f"{target.name()}.{self._attrs['op']}.gen_function"
        func = registry.get(func_key)
        return func(self._attrs)
