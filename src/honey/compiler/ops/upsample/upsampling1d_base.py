import itertools
import logging
import re
from collections import OrderedDict
from typing import List

import jinja2

from honey import backend
from honey.backend import registry
from honey.compiler.base import Operator, Tensor
from honey.utils import shape_utils

# pylint: disable=C0103,W0221,R1732,W0613
logging.basicConfig(level=logging.INFO)

SHAPE_FUNC_TEMPLATE = jinja2.Template(
    """
{{indent}}{{dtype}}NI = {{x_dim0}};
{{indent}}{{dtype}}WI = {{x_dim1}};
{{indent}}{{dtype}}CI = {{x_dim2}};
{{indent}}{{dtype}}CO = {{x_dim2}};
{{indent}}{{dtype}}NO = NI;
{{indent}}{{dtype}}WO = WI * {{scale_factor}};
"""
)
_SHAPE_FUNC_TEMPLATE = jinja2.Template(
    """
{{indent}}{{dtype}}NI = {{x_dim0}};
{{indent}}{{dtype}}WI = {{x_dim1}};
{{indent}}{{dtype}}CI = {{x_dim2}};
{{indent}}{{dtype}}CO = {{x_dim2}};
{{indent}}{{dtype}}NO = NI;
{{indent}}{{dtype}}WO = {{out_w}};
"""
)

SHAPE_ASSIGNMENT_TEMPLATE = jinja2.Template(
    """
{{indent}}{{y_dim0}} = NO;
{{indent}}{{y_dim1}} = WO;
"""
)

EXEC_COND_TEMPLATE = jinja2.Template(
    """
{{indent}}if ({{cond}}) {
{{indent}}  {{program}}
{{indent}}}
"""
)


class upsampling1d_base(Operator):
    """
    Upsamples a given multi-channel 1D (spatial) data.

    * :attr:`scale_factor` (float): multiplier for spatial size.

    * :attr:`mode` (str): the upsampling algorithm: one of ``'nearest'``,
      ``'nearest-exact'``, ``'linear'``, ``'bilinear'``, and ``'area'``.
      Currently we support ``'linear'``,  ``'nearest-exact'``, and ``'nearest'`` mode.

    Args:
        input (Tensor [N, W, C]): the input data.

    Return:
        Tensor [N, W_out, C].
    """

    def __init__(self, scale_factor, mode, align_corners=False) -> None:
        super().__init__()
        self._attrs["op"] = "upsampling1d"
        self._attrs["scale_factor"] = scale_factor
        self._attrs["align_corners"] = align_corners
        self._attrs["mode"] = mode
        self._attrs["out_shape"] = False
        self.shape_eval_template = SHAPE_FUNC_TEMPLATE
        self.shape_save_template = SHAPE_ASSIGNMENT_TEMPLATE
        self.exec_cond_template = EXEC_COND_TEMPLATE

    def _infer_shape(self, x: List[int], out: List[int] = None):
        self.shape_eval_template = (
            SHAPE_FUNC_TEMPLATE if out is None else _SHAPE_FUNC_TEMPLATE
        )
        args = {
            "indent": "",
            "dtype": "",
            "div": "//",
            "x_dim0": x[0],
            "x_dim1": x[1],
            "x_dim2": x[2],
        }
        if out is None:
            args["scale_factor"] = self._attrs["scale_factor"]
        else:
            args["out_w"] = out[1]
        self.shape_args = args
        eval_func = self.shape_eval_template.render(**args)
        output = {}
        exec(eval_func, output)  # noqa: P204
        return [
            int(output["NO"]),
            int(output["WO"]),
            int(output["CO"]),
        ]

    def _infer_shapes(self, x: Tensor, out: Tensor = None):
        x_shape_values = [var._attrs["values"] for var in x._attrs["shape"]]
        x_shapes = itertools.product(*x_shape_values)
        # run infershape for each
        if out is None:
            out_shapes = [None] * len(x.shape())
        else:
            out_shape_values = [var._attrs["values"] for var in out._attrs["shape"]]
            out_shapes = itertools.product(*out_shape_values)
        y_shapes = []
        for x_shape, out_shape in zip(x_shapes, out_shapes):
            y_shape = self._infer_shape(x_shape, out_shape)
            y_shapes.append(y_shape)

        def unique(vector):
            return sorted(set(vector))

        output_shape = [
            x.shape()[0],
            shape_utils.gen_int_var(unique([d[1] for d in y_shapes])),
            shape_utils.gen_int_var(unique([d[2] for d in y_shapes])),
        ]

        in_w = x._attrs["shape"][1]._attrs["symbolic_value"]
        out_w = (
            in_w * int(self._attrs["scale_factor"])
            if out is None
            else out._attrs["shape"][1]._attrs["symbolic_value"]
        )

        output_shape[1]._attrs["symbolic_value"] = out_w

        return output_shape

    def _invert_exec_key(self, key):
        tmp = re.findall(r"(\d+)", key)
        return [int(x) for x in tmp]

    def _gen_exec_key(self, shape: List[int]):
        return self.exec_key_template.render(
            x_dim0=shape[0], x_dim1=shape[1], x_dim2=shape[2]
        ).replace("\n", "")

    def _extract_exec_path(self, x: Tensor):
        self._attrs["exec_path"] = OrderedDict()
        self._attrs["exec_path"]["true"] = ""

    def _signature(self):
        signature = "upsampling1d: S=[{s}], M=[{m}]]".format(
            s=self._attrs["scale_factor"], m=self._attrs["mode"]
        )
        return signature

    def __call__(self, x: Tensor, out: Tensor = None) -> List[Tensor]:
        self._attrs["out_shape"] = True if out is not None else False
        self._attrs["inputs"] = [x]
        self._set_depth()
        self._extract_exec_path(x)
        output_shape = self._infer_shapes(x, out)
        output = Tensor(output_shape, src_ops={self}, dtype=x._attrs["dtype"])
        self._attrs["outputs"] = [output]
        return output

    def _get_op_attributes(self):
        return {
            "mode": self._attrs["mode"],
            "scale_factor": self._attrs["scale_factor"],
            "out_shape": self._attrs["out_shape"],
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
            self.exec_cond_template,
            self.shape_eval_template,
            self.shape_save_template,
        )
