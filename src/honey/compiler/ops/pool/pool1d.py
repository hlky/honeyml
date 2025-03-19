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
{{indent}}{{dtype}}KW = {{kernel_w}};
{{indent}}{{dtype}}SW = {{stride}};
{{indent}}{{dtype}}PW = {{pad}};
{{indent}}{{dtype}}NO = NI;
{{indent}}{{dtype}}WO = (WI + PW + PW - KW) {{div}} SW + 1;
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


class pool1d_base(Operator):
    """Base class of pool1d."""

    def __init__(self, stride, pad, kernel_size, reduce_func) -> None:
        """
        Parameters
        ----------
        stride : int
        pad : int
        reduce_func : the function to use for reduction
        """
        super().__init__()
        self._attrs["op"] = "pool1d"
        self._attrs["stride"] = stride
        self._attrs["pad"] = pad
        self._attrs["reduce_func"] = reduce_func
        self._attrs["kernel_size"] = kernel_size
        self._attrs["KW"] = kernel_size
        self.shape_eval_template = SHAPE_FUNC_TEMPLATE
        self.shape_save_template = SHAPE_ASSIGNMENT_TEMPLATE
        self.exec_cond_template = EXEC_COND_TEMPLATE

    def _infer_shape(self, x: List[int]):
        eval_func = self.shape_eval_template.render(
            indent="",
            dtype="",
            div="//",
            stride=self._attrs["stride"],
            pad=self._attrs["pad"],
            x_dim0=x[0],
            x_dim1=x[1],
            x_dim2=x[2],
            kernel_w=self._attrs["kernel_size"],
        )
        output = {}
        exec(eval_func, output)  # noqa: P204
        return [
            int(output["NO"]),
            int(output["WO"]),
            int(output["CO"]),
        ]

    def _infer_shapes(self, x: Tensor):
        x_shape_values = [var._attrs["values"] for var in x._attrs["shape"]]
        x_shapes = itertools.product(*x_shape_values)
        # run infershape for each
        y_shapes = []
        for x_shape in x_shapes:
            y_shape = self._infer_shape(x_shape)
            y_shapes.append(y_shape)

        def unique(vector):
            return sorted(set(vector))

        output_shape = [
            x._attrs["shape"][0],
            shape_utils.gen_int_var(unique([d[1] for d in y_shapes])),
            shape_utils.gen_int_var(unique([d[2] for d in y_shapes])),
        ]
        return output_shape

    def _invert_exec_key(self, key: str):
        tmp = re.findall(r"(\d+)", key)
        return [int(x) for x in tmp]

    def _gen_exec_key(self, shape):
        return self.exec_key_template.render(
            x_dim0=shape[0], x_dim1=shape[1], x_dim2=shape[2]
        ).replace("\n", "")

    def _extract_exec_path(self, x: Tensor):
        self._attrs["exec_path"] = OrderedDict()
        self._attrs["exec_path"]["true"] = ""

    def _signature(self) -> str:
        signature = "pooling1d: K=[{kw}], S=[{s}], P=[{p}], CO=[{co}]".format(
            kw=self._attrs["KW"],
            s=self._attrs["stride"],
            p=self._attrs["pad"],
            co=self._attrs["CO"],
        )
        return signature

    def __call__(self, x: Tensor) -> List[Tensor]:
        self._attrs["inputs"] = [x]
        self._set_depth()
        self._extract_exec_path(x)
        output_shape = self._infer_shapes(x)
        output = Tensor(output_shape, src_ops={self}, dtype=x._attrs["dtype"])
        self._attrs["outputs"] = [output]
        return output

    def _get_op_attributes(self):
        target_attrs = ["stride", "pad", "kernel_size", "reduce_func"]
        attr = {}

        for target_attr in target_attrs:
            if target_attr in self._attrs:
                attr[target_attr] = self._attrs[target_attr]

        return attr

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = "{target}.{op}.gen_function".format(
            target=target.name(), op=self._attrs["op"]
        )
        func = registry.get(func_key)
        return func(
            self._attrs,
            self.exec_cond_template,
            self.shape_eval_template,
            self.shape_save_template,
        )
