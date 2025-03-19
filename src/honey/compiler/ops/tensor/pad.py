from typing import List, Tuple, Union

from honey import backend
from honey.backend import registry
from honey.compiler.base import Operator, Tensor


class pad(Operator):
    """
    Pads a tensor according to the specified padding mode and padding values.

    Args:
        pad (Tuple[int]): Padding on each dimension.
        mode (str): Padding mode, one of 'constant', 'reflect', 'replicate', 'circular'.
        value (float, optional): Fill value for 'constant' padding mode. Default is 0.
    """

    def __init__(
        self, pad: Union[int, Tuple[int]], mode: str = "constant", value: float = 0.0
    ) -> None:
        super().__init__()
        self._attrs["op"] = "pad"
        self._attrs["pad"] = pad
        self._attrs["mode"] = mode
        self._attrs["value"] = value

    def _infer_shape(self, x: List[int]):
        pad = self._attrs["pad"]
        rank = len(x)
        if rank == 1:
            output_shape = [x[0] + pad[0] + pad[1]]
        elif rank == 2:
            output_shape = [x[0], x[1] + pad[0] + pad[1]]
        elif rank == 3:
            output_shape = [x[0], x[1] + pad[2] + pad[3], x[2] + pad[0] + pad[1]]
        elif rank == 4:
            output_shape = [x[0], x[1] + pad[2] + pad[3], x[2] + pad[0] + pad[1], x[3]]
        elif rank == 5:
            output_shape = [x[0], x[1] + pad[4] + pad[5], x[2] + pad[2] + pad[3], x[3] + pad[0] + pad[1], x[4]]
        else:
            raise NotImplementedError(f"unsupported rank {rank}")
        return output_shape

    def _signature(self):
        return f"pad: {self._attrs['mode']}"

    def __call__(self, x: Tensor) -> Tensor:
        rank = x._rank()
        if isinstance(self._attrs["pad"], int):
            repeat = 2
            if rank == 3:
                repeat = 4
            if rank == 4:
                repeat = 4
            if rank == 5:
                repeat = 5
            self._attrs["pad"] = (self._attrs["pad"],) * repeat
        if rank == 4 and len(self._attrs["pad"]) > 4:
            raise NotImplementedError("padding channels of 4D tensor not implemented")
        self._attrs["inputs"] = [x]
        self._set_depth()
        output_shape = self._infer_shape(x._attrs["shape"])
        output = Tensor(output_shape, src_ops={self}, dtype=x._attrs["dtype"])
        self._attrs["outputs"] = [output]
        return output

    def _get_op_attributes(self):
        return {
            "pad": self._attrs["pad"],
            "mode": self._attrs["mode"],
            "value": self._attrs["value"],
        }

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
