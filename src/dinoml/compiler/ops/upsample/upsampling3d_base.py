import itertools
import logging
import re
from collections import OrderedDict
from typing import List

import jinja2

from dinoml import backend
from dinoml.backend import registry
from dinoml.compiler.base import Operator, Tensor
from dinoml.utils import shape_utils

logging.basicConfig(level=logging.INFO)

SHAPE_FUNC_TEMPLATE = jinja2.Template(
    """
{{indent}}{{dtype}}NI = {{x_dim0}};
{{indent}}{{dtype}}FI = {{x_dim1}};
{{indent}}{{dtype}}HI = {{x_dim2}};
{{indent}}{{dtype}}WI = {{x_dim3}};
{{indent}}{{dtype}}CI = {{x_dim4}};
{{indent}}{{dtype}}CO = {{x_dim4}};
{{indent}}{{dtype}}NO = NI;
{{indent}}{{dtype}}FO = FI * {{scale_factor}};
{{indent}}{{dtype}}HO = HI * {{scale_factor}};
{{indent}}{{dtype}}WO = WI * {{scale_factor}};
"""
)

_SHAPE_FUNC_TEMPLATE = jinja2.Template(
    """
{{indent}}{{dtype}}NI = {{x_dim0}};
{{indent}}{{dtype}}FI = {{x_dim1}};
{{indent}}{{dtype}}HI = {{x_dim2}};
{{indent}}{{dtype}}WI = {{x_dim3}};
{{indent}}{{dtype}}CI = {{x_dim4}};
{{indent}}{{dtype}}CO = {{x_dim4}};
{{indent}}{{dtype}}NO = NI;
{{indent}}{{dtype}}FO = {{out_f}};
{{indent}}{{dtype}}HO = {{out_h}};
{{indent}}{{dtype}}WO = {{out_w}};
"""
)

SHAPE_ASSIGNMENT_TEMPLATE = jinja2.Template(
    """
{{indent}}{{y_dim0}} = NO;
{{indent}}{{y_dim1}} = FO;
{{indent}}{{y_dim2}} = HO;
{{indent}}{{y_dim3}} = WO;
"""
)

EXEC_COND_TEMPLATE = jinja2.Template(
    """
{{indent}}if ({{cond}}) {
{{indent}}  {{program}}
{{indent}}}
"""
)


class upsampling3d_base(Operator):
    """
    Upsamples a given multi-channel 3D (spatiotemporal/volumetric) data.

    Layout: [N, F, H, W, C] (channels last)

    Attributes:
      * scale_factor (float/int): multiplier for F/H/W
      * mode (str): 'trilinear', 'nearest', 'nearest-exact'
      * align_corners (bool): for trilinear only (ignored otherwise)

    Args:
        input (Tensor [N, F, H, W, C])

    Returns:
        Tensor [N, F_out, H_out, W_out, C]
    """

    def __init__(self, scale_factor, mode, align_corners=False) -> None:
        super().__init__()
        self._attrs["op"] = "upsampling3d"
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
            "x_dim3": x[3],
            "x_dim4": x[4],
        }
        if out is None:
            args["scale_factor"] = self._attrs["scale_factor"]
        else:
            args["out_f"] = out[1]
            args["out_h"] = out[2]
            args["out_w"] = out[3]

        self.shape_args = args
        eval_func = self.shape_eval_template.render(**args)
        output = {}
        exec(eval_func, output)  # noqa: P204
        return [
            int(output["NO"]),
            int(output["FO"]),
            int(output["HO"]),
            int(output["WO"]),
            int(output["CO"]),
        ]

    def _infer_shapes(self, x: Tensor, out: Tensor = None):
        x_shape_values = [var._attrs["values"] for var in x._attrs["shape"]]
        x_shapes = itertools.product(*x_shape_values)

        if out is None:
            out_shapes = [None] * len(x.shape())
        else:
            out_shape_values = [var._attrs["values"] for var in out._attrs["shape"]]
            out_shapes = itertools.product(*out_shape_values)

        y_shapes = []
        for x_shape, out_shape in zip(x_shapes, out_shapes):
            y_shapes.append(self._infer_shape(x_shape, out_shape))

        def unique(vector):
            return sorted(set(vector))

        output_shape = [
            x.shape()[0],
            shape_utils.gen_int_var(unique([d[1] for d in y_shapes])),
            shape_utils.gen_int_var(unique([d[2] for d in y_shapes])),
            shape_utils.gen_int_var(unique([d[3] for d in y_shapes])),
            shape_utils.gen_int_var(unique([d[4] for d in y_shapes])),
        ]

        in_f = x._attrs["shape"][1]
        in_h = x._attrs["shape"][2]
        in_w = x._attrs["shape"][3]

        out_f = (
            in_f * int(self._attrs["scale_factor"])
            if out is None
            else out._attrs["shape"][1]
        )
        out_h = (
            in_h * int(self._attrs["scale_factor"])
            if out is None
            else out._attrs["shape"][2]
        )
        out_w = (
            in_w * int(self._attrs["scale_factor"])
            if out is None
            else out._attrs["shape"][3]
        )

        output_shape[1] = out_f
        output_shape[2] = out_h
        output_shape[3] = out_w

        return output_shape

    def _invert_exec_key(self, key):
        tmp = re.findall(r"(\d+)", key)
        return [int(v) for v in tmp]

    def _gen_exec_key(self, shape: List[int]):
        return self.exec_key_template.render(
            x_dim0=shape[0],
            x_dim1=shape[1],
            x_dim2=shape[2],
            x_dim3=shape[3],
            x_dim4=shape[4],
        ).replace("\n", "")

    def _extract_exec_path(self, x: Tensor):
        self._attrs["exec_path"] = OrderedDict()
        self._attrs["exec_path"]["true"] = ""

    def _signature(self):
        return "upsampling3d: S=[{s}], M=[{m}]".format(
            s=self._attrs["scale_factor"], m=self._attrs["mode"]
        )

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
            "align_corners": self._attrs["align_corners"],
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
