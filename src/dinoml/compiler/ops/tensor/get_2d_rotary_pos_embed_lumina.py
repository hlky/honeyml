from dinoml import backend
from dinoml.backend import registry
from dinoml.compiler.base import IntImm, Operator, Tensor
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
        len_h: int,
        len_w: int,
        linear_factor: float = 1.0,
        ntk_factor: float = 1.0,
        dtype: str = "float32",
    ):
        self._attrs["embed_dim"] = embed_dim
        self._attrs["len_h"] = len_h
        self._attrs["len_w"] = len_w
        self._attrs["linear_factor"] = float(linear_factor)
        self._attrs["ntk_factor"] = float(ntk_factor)
        self._attrs["inputs"] = []
        self._attrs["dtype"] = normalize_dtype(dtype)

        self._set_depth()

        if isinstance(embed_dim, int):
            out_cols = int(embed_dim) // 2
        else:
            out_cols = embed_dim / 2

        real = Tensor(
            [len_h, len_w, out_cols],
            src_ops={self},
            dtype=self._attrs["dtype"],
            skip_constant_folding=True,
        )
        imag = Tensor(
            [len_h, len_w, out_cols],
            src_ops={self},
            dtype=self._attrs["dtype"],
            skip_constant_folding=True,
        )

        self._attrs["outputs"] = [real, imag]
        return real, imag

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = f"{target.name()}.{self._attrs['op']}.gen_function"
        func = registry.get(func_key)
        return func(self._attrs)
