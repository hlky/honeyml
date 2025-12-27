import itertools
from typing import List

import jinja2

from dinoml import backend
from dinoml.backend import registry
from dinoml.compiler.base import Operator, Tensor
from dinoml.utils import shape_utils

SHAPE_FUNC_TEMPLATE = jinja2.Template(
    """
{{indent}}{{dtype}}NI = {{x_dim0}};
{{indent}}{{dtype}}HI = {{x_dim1}};
{{indent}}{{dtype}}WI = {{x_dim2}};
{{indent}}{{dtype}}CI = {{x_dim3}};
{{indent}}{{dtype}}CO = CI * ({{r}} * {{r}});
{{indent}}{{dtype}}HO = HI / {{r}};
{{indent}}{{dtype}}WO = WI / {{r}};
"""
)


class pixel_unshuffle(Operator):
    """
    Rearranges elements in a tensor of shape [N, H * r, W * r, C] to a tensor of shape [N, H, W, C * r^2].

    * :attr:`r` (int): downscale factor.

    Args:
        input (Tensor [N, H * r, W * r, C]): the input data.

    Return:
        Tensor [N, H, W, C * r^2].
    """

    def __init__(self, r) -> None:
        super().__init__()
        self._attrs["op"] = "pixel_unshuffle"
        self._attrs["r"] = r
        self.shape_eval_template = SHAPE_FUNC_TEMPLATE

    def _infer_shape(self, x: List[int]):
        args = {
            "indent": "",
            "dtype": "",
            "x_dim0": x[0],
            "x_dim1": x[1],
            "x_dim2": x[2],
            "x_dim3": x[3],
            "r": self._attrs["r"],
        }
        eval_func = self.shape_eval_template.render(**args)
        output = {}
        exec(eval_func, output)  # noqa: P204
        return [
            int(output["NI"]),
            int(output["HO"]),
            int(output["WO"]),
            int(output["CO"]),
        ]

    def _infer_shapes(self, x: Tensor):
        x_shape_values = [var._attrs["values"] for var in x._attrs["shape"]]
        x_shapes = itertools.product(*x_shape_values)
        y_shapes = [self._infer_shape(x_shape) for x_shape in x_shapes]

        def unique(vector):
            return sorted(set(vector))

        output_shape = [
            x.shape()[0],
            shape_utils.gen_int_var(unique([d[1] for d in y_shapes])),
            shape_utils.gen_int_var(unique([d[2] for d in y_shapes])),
            shape_utils.gen_int_var(unique([d[3] for d in y_shapes])),
        ]

        return output_shape

    def _signature(self):
        signature = "pixel_unshuffle: r=[{r}]".format(r=self._attrs["r"])
        return signature

    def __call__(self, x: Tensor) -> List[Tensor]:
        self._attrs["inputs"] = [x]
        self._set_depth()
        output_shape = self._infer_shapes(x)
        output = Tensor(output_shape, src_ops={self}, dtype=x._attrs["dtype"])
        self._attrs["outputs"] = [output]
        return output

    def _get_op_attributes(self):
        return {
            "r": self._attrs["r"],
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
