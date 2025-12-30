from dinoml import backend
from dinoml.backend import registry
from dinoml.compiler.base import IntVar, Operator, Tensor
from dinoml.compiler.dtype import normalize_dtype


class cropped_pos_embed(Operator):
    def __init__(self):
        super().__init__()
        self._attrs["op"] = "cropped_pos_embed"
        self._attrs["has_profiler"] = False
        self._attrs["nop"] = False

    def __call__(
        self,
        embed_dim: int,
        pos_embed_max_size: int,
        base_size: int,
        interpolation_scale: float,
        patch_size: int,
        height: int,
        width: int,
        dtype: str = "float32",
    ) -> Tensor:
        if embed_dim % 2 != 0:
            raise ValueError("embed_dim must be divisible by 2")
        if pos_embed_max_size is None:
            raise ValueError("`pos_embed_max_size` must be set for cropping.")

        if isinstance(height, IntVar):
            Hc: IntVar = height / patch_size
            Hc_check = Hc.upper_bound()
        else:
            Hc = height // patch_size
            Hc_check = Hc
        if isinstance(width, IntVar):
            Wc: IntVar = width / patch_size
            Wc_check = Wc.upper_bound()
        else:
            Wc = width // patch_size
            Wc_check = Wc

        if Hc_check > pos_embed_max_size:
            raise ValueError(
                f"Height ({Hc}) cannot be greater than `pos_embed_max_size`: {pos_embed_max_size}."
            )
        if Wc_check > pos_embed_max_size:
            raise ValueError(
                f"Width ({Wc}) cannot be greater than `pos_embed_max_size`: {pos_embed_max_size}."
            )

        self._attrs["inputs"] = []
        self._attrs["dtype"] = normalize_dtype(dtype)

        # keep same argument names as the python reference
        self._attrs["embed_dim"] = int(embed_dim)
        self._attrs["pos_embed_max_size"] = int(pos_embed_max_size)
        self._attrs["base_size"] = int(base_size)
        self._attrs["interpolation_scale"] = float(interpolation_scale)
        self._attrs["patch_size"] = int(patch_size)
        self._attrs["height"] = height
        self._attrs["width"] = width

        self._set_depth()

        # output shape: [1, Hc*Wc, D]
        y = Tensor(
            [1, Hc * Wc, embed_dim],
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
