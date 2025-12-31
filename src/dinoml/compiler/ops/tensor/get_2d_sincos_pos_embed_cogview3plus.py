from dinoml import backend
from dinoml.backend import registry
from dinoml.compiler.base import IntImm, Operator, Tensor
from dinoml.compiler.dtype import normalize_dtype


class get_2d_sincos_pos_embed_cogview3plus(Operator):
    """
    CogView3Plus fused op to build the final positional embedding tensor:

      image_pos_embed = pos_table[:height, :width].reshape(height*width, hidden_size)
      text_pos_embed = zeros([text_length, hidden_size])
      out = cat([text_pos_embed, image_pos_embed], dim=0)[None, ...]

    Inputs:
      - pos_table: Tensor [pos_embed_max_size, pos_embed_max_size, hidden_size]

    Output:
      - Tensor [1, text_length + height*width, hidden_size]
    """

    def __init__(self):
        super().__init__()
        self._attrs["op"] = "get_2d_sincos_pos_embed_cogview3plus"
        self._attrs["has_profiler"] = False
        self._attrs["nop"] = False

    def __call__(
        self,
        pos_table: Tensor,
        hidden_size: int,
        pos_embed_max_size: int,
        height: int,
        width: int,
        text_length: int,
        dtype: str = None,
    ) -> Tensor:
        # dtype defaults to pos_table dtype
        if dtype is None:
            dtype = pos_table._attrs["dtype"]

        self._attrs["inputs"] = [pos_table]
        self._attrs["dtype"] = normalize_dtype(dtype)

        self._attrs["hidden_size"] = hidden_size
        self._attrs["pos_embed_max_size"] = int(pos_embed_max_size)
        self._attrs["height"] = height
        self._attrs["width"] = width
        self._attrs["text_length"] = text_length

        self._set_depth()

        total_len = text_length + height * width

        y = Tensor(
            [1, total_len, hidden_size],
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
