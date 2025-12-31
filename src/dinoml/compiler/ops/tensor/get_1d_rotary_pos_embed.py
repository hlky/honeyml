from typing import Optional, Union

from dinoml import backend
from dinoml.backend import registry
from dinoml.compiler.base import IntImm, IntVar, Operator, Tensor
from dinoml.compiler.dtype import normalize_dtype


class get_1d_rotary_pos_embed(Operator):
    """
    DinoML op version of get_1d_rotary_pos_embed.

    Supports:
      - pos: int (treated as torch.arange(pos)) OR a 1D Tensor [S]
      - use_real=False: returns (cis_real, cis_imag) each [S, dim/2]
      - use_real=True: returns (freqs_cos, freqs_sin) each [S, dim]
        * repeat_interleave_real=True: repeat_interleave(2)
        * repeat_interleave_real=False: cat([x, x])
    """

    def __init__(self):
        super().__init__()
        self._attrs["op"] = "get_1d_rotary_pos_embed"
        self._attrs["has_profiler"] = False
        self._attrs["nop"] = False

    def __call__(
        self,
        dim: int,
        pos: Union[Tensor, int],
        theta: float = 10000.0,
        use_real: bool = False,
        linear_factor: float = 1.0,
        ntk_factor: float = 1.0,
        repeat_interleave_real: bool = True,
        dtype: str = "float32",
    ):
        if dim % 2 != 0:
            raise ValueError("`dim` must be divisible by 2")

        self._attrs["dim"] = dim
        self._attrs["theta"] = float(theta)
        self._attrs["use_real"] = bool(use_real)
        self._attrs["linear_factor"] = float(linear_factor)
        self._attrs["ntk_factor"] = float(ntk_factor)
        self._attrs["repeat_interleave_real"] = bool(repeat_interleave_real)

        self._attrs["dtype"] = normalize_dtype(dtype)

        if isinstance(pos, Tensor):
            self._attrs["inputs"] = [pos]
            self._attrs["pos_is_tensor"] = True
            self._attrs["pos_int"] = 0
            S_dim = pos._attrs["shape"][0]
        elif isinstance(pos, int):
            self._attrs["inputs"] = []
            self._attrs["pos_is_tensor"] = False
            self._attrs["pos_int"] = pos
            S_dim = pos
        else:
            raise RuntimeError(f"`pos` must be a Tensor or int, got {type(pos)}")

        self._set_depth()

        if isinstance(dim, (IntImm, IntVar)):
            out_cols = dim if use_real else (dim / 2)
        else:
            out_cols = dim if use_real else (dim // 2)

        out0 = Tensor(
            [S_dim, out_cols],
            src_ops={self},
            dtype=self._attrs["dtype"],
            skip_constant_folding=True,
        )
        out1 = Tensor(
            [S_dim, out_cols],
            src_ops={self},
            dtype=self._attrs["dtype"],
            skip_constant_folding=True,
        )

        self._attrs["outputs"] = [out0, out1]
        return out0, out1

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = f"{target.name()}.{self._attrs['op']}.gen_function"
        func = registry.get(func_key)
        return func(self._attrs)
