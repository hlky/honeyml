#  Copyright 2025 hlky. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""
Common NHWC padding ops
"""
import itertools
from typing import List

import jinja2

from honey import backend
from honey.backend import registry
from honey.compiler.base import Operator, Tensor
from honey.utils import shape_utils

# pylint: disable=C0103,W0221


SHAPE_ASSIGNMENT_TEMPLATE = jinja2.Template(
    """
{{indent}}{{y_dim0}} = NO;
{{indent}}{{y_dim1}} = HO;
{{indent}}{{y_dim2}} = WO;
"""
)


class nhwc_pad_common(Operator):
    """
    Pad the 3-channel input data to 4/8-channel.
    """

    def __init__(self, shape_func_template, padded_channels):
        super().__init__()
        self._attrs["op"] = f"nhwc3to{padded_channels}"
        self.shape_eval_template = shape_func_template
        self.shape_save_template = SHAPE_ASSIGNMENT_TEMPLATE

    def _infer_shape(self, x: List[int]):
        eval_func = self.shape_eval_template.render(
            indent="",
            dtype="",
            x_dim0=x[0],
            x_dim1=x[1],
            x_dim2=x[2],
            x_dim3=x[3],
        )
        output = {}
        exec(eval_func, output)  # noqa: P204
        return [
            int(output["NO"]),
            int(output["HO"]),
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
            shape_utils.gen_int_var(unique([d[0] for d in y_shapes])),
            shape_utils.gen_int_var(unique([d[1] for d in y_shapes])),
            shape_utils.gen_int_var(unique([d[2] for d in y_shapes])),
            shape_utils.gen_int_var(unique([d[3] for d in y_shapes])),
        ]
        return output_shape

    def __call__(self, x: Tensor) -> List[Tensor]:
        self._attrs["inputs"] = [x]
        self._set_depth()
        output_shape = self._infer_shapes(x)
        output = Tensor(output_shape, src_ops={self}, dtype=x.dtype())
        self._attrs["outputs"] = [output]
        return output

    def _get_op_attributes(self):
        return {
            "padded_channels": self._attrs["op"].split("to")[-1],
            "shape_func_template": self.shape_eval_template,
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
            self.shape_eval_template,
            self.shape_save_template,
        )
