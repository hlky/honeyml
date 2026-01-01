from dinoml import backend
from dinoml.backend import registry
from dinoml.compiler.base import IntImm, Operator, Tensor
from dinoml.compiler.dtype import normalize_dtype


class gaussian_fourier_projection(Operator):
    """
    DinoML op version of GaussianFourierProjection.forward:

      if log: x = log(x)
      x_proj = x[:,None] * weight[None,:] * 2*pi
      out = cat([sin(x_proj), cos(x_proj)], dim=-1) or flipped [cos, sin]

    Inputs:
      x: Tensor [N]
      weight: Tensor [E]
    Output:
      y: Tensor [N, 2E]
    """

    def __init__(self):
        super().__init__()
        self._attrs["op"] = "gaussian_fourier_projection"
        self._attrs["has_profiler"] = False
        self._attrs["nop"] = False

    def __call__(
        self, x: Tensor, weight: Tensor, log: bool = True, flip_sin_to_cos: bool = False
    ) -> Tensor:
        self._attrs["inputs"] = [x, weight]
        self._attrs["dtype"] = x._attrs[
            "dtype"
        ]  # output dtype follows x (common usage)
        self._attrs["log"] = bool(log)
        self._attrs["flip_sin_to_cos"] = bool(flip_sin_to_cos)

        # embedding_size from weight shape[0]
        if len(weight._attrs["shape"]) != 1:
            raise ValueError("`weight` must be a 1D tensor [embedding_size]")
        emb = weight._attrs["shape"][0]
        self._attrs["embedding_size"] = emb

        self._set_depth()

        # x is expected 1D
        if len(x._attrs["shape"]) != 1:
            raise ValueError("`x` must be a 1D tensor [N]")

        n = x._attrs["shape"][0]
        y = Tensor(
            [n, emb * 2],
            src_ops={self},
            dtype=normalize_dtype(self._attrs["dtype"]),
        )
        self._attrs["outputs"] = [y]
        return y

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = f"{target.name()}.{self._attrs['op']}.gen_function"
        func = registry.get(func_key)
        return func(self._attrs)
