from dinoml import backend
from dinoml.backend import registry
from dinoml.compiler.base import Operator, Tensor
from dinoml.compiler.dtype import normalize_dtype


class get_fourier_embeds_from_boundingbox(Operator):
    def __init__(self):
        super().__init__()
        self._attrs["op"] = "get_fourier_embeds_from_boundingbox"
        self._attrs["has_profiler"] = False
        self._attrs["nop"] = False

    def __call__(self, embed_dim: int, box: Tensor) -> Tensor:
        if len(box._attrs["shape"]) != 3:
            raise ValueError("`box` must be a 3D tensor [B, N, 4]")

        self._attrs["inputs"] = [box]
        self._attrs["dtype"] = normalize_dtype(box._attrs["dtype"])
        self._attrs["embed_dim"] = embed_dim

        self._set_depth()

        B = box._attrs["shape"][0]
        N = box._attrs["shape"][1]
        out_last = embed_dim * 8

        y = Tensor(
            [B, N, out_last],
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
