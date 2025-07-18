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
Permute(0, 2, 1, 3) op.
Change the dimensions dim1 and dim2 of input 4d tensor.
"""
from typing import List

from honey import backend

from honey.backend import registry
from honey.compiler.base import IntVar, Operator, Tensor

# pylint: disable=C0103,W0221


class permute0213(Operator):
    """
    Permutes the input 4d tensor from (B, N, M, K) to (B, M, N, K).

    Args:
        input (Tensor[B, N, M, K]): the source tensor with 3 dimensions

    Returns:
        output (Tensor[B, M, N, K]): the destination tensor

    Example:

        .. highlight:: python
        .. code-block:: python

            X = Tensor(shape=[2, 384, 262, 10], name="X", is_input=True)
            Y = ops.permute0213()(X)
            y_shape = [d._attrs["values"][0] for d in Y.shape()]
            print(y_shape)

            Outs:
            [2, 262, 384, 10]

    """

    def __init__(self):
        super().__init__()
        self._attrs["op"] = "permute0213"

    def _infer_shapes(self, x: Tensor) -> List[IntVar]:
        """Infers shapes for permute0213."""

        x_shape = x._attrs["shape"]
        return [x_shape[0], x_shape[2], x_shape[1], x_shape[3]]

    def __call__(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor

        Returns
        -------
        Tensor
            Generate output tensors of function calls.
            In permute0213, its a 4d tensor with d0,d2,d1,d3 of
            input Tensor.
        """
        self._attrs["inputs"] = [x]
        self._set_depth()
        output_shape = self._infer_shapes(x)
        output = Tensor(output_shape, src_ops={self})
        output._attrs["dtype"] = x.dtype()
        self._attrs["outputs"] = [output]
        return output

    def gen_function(self) -> str:
        """Generate function body."""
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
