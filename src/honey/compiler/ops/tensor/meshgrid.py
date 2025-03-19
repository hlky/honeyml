from typing import List

from honey import backend
from honey.backend import registry
from honey.compiler.base import Operator, Tensor


class meshgrid(Operator):
    """
    Creates coordinate matrices from coordinate vectors.

    Args:
        input (List[Tensor]): the input coordinate vectors.
        indexing (str): 'ij' or 'xy'. Default is 'ij'.

    Return:
        List[Tensor]: the output coordinate matrices.
    """

    def __init__(self, indexing="ij") -> None:
        super().__init__()
        self._attrs["op"] = "meshgrid"
        self._attrs["indexing"] = indexing

    def _infer_shapes(self, inputs: List[Tensor]):
        shapes = [tensor._attrs["shape"][0] for tensor in inputs]
        output_shapes = [list(shapes) for _ in shapes]
        if self._attrs["indexing"] == "xy":
            output_shapes[0], output_shapes[1] = output_shapes[1], output_shapes[0]
        return output_shapes

    def _signature(self):
        return f"meshgrid: {self._attrs['indexing']}"

    def __call__(self, *inputs: Tensor) -> List[Tensor]:
        self._attrs["inputs"] = list(inputs)
        self._set_depth()
        output_shapes = self._infer_shapes(inputs)
        outputs = [
            Tensor(shape, src_ops={self}, dtype=inputs[0]._attrs["dtype"])
            for shape in output_shapes
        ]
        self._attrs["outputs"] = outputs
        return outputs

    def _get_op_attributes(self):
        return {"indexing": self._attrs["indexing"]}

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        template_path = target.template_path()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(
            self._attrs,
            template_path,
        )
