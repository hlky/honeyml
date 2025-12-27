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
activation modules.
"""

from dinoml.compiler import ops
from dinoml.compiler.base import Tensor
from dinoml.compiler.public import elementwise, FuncEnum
from dinoml.frontend.nn.module import Module


class GELU(Module):
    r"""Applies the Gaussian Error Linear Units function:

    .. math:: \text{GELU}(x) = x * \Phi(x)

    where :math:`\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.

    When the approximate argument is 'tanh', Gelu is estimated with:

    .. math:: \text{GELU}(x) = 0.5 * x * (1 + \text{Tanh}(\sqrt(2 / \pi) * (x + 0.044715 * x^3)))

    Args:
        approximate (str, optional): the gelu approximation algorithm to use:
            ``'none'`` | ``'tanh'``. Default: ``'none'``
    """

    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate

    def forward(self, *args):
        assert len(args) == 1
        input_val = args[0]

        # For extra speedup, lower to fast_gelu
        if self.approximate == "tanh":
            result = elementwise(FuncEnum.FASTGELU)(input_val)
        else:
            result = elementwise(FuncEnum.GELU)(input_val)

        return result


class LeakyReLU(Module):
    def __init__(self, negative_slope: float) -> None:
        super().__init__()
        self.negative_slope = negative_slope
        self.op = ops.leaky_relu

    def forward(self, tensor: Tensor) -> Tensor:
        out = self.op(tensor, self.negative_slope)
        return out
