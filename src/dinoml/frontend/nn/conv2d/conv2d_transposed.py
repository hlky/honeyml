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
common module for ConvTranspose2d subgraph
"""

from dinoml.compiler import ops
from dinoml.compiler.base import Tensor
from dinoml.frontend.nn.module import Module
from dinoml.frontend.nn.parameter import Parameter

# pylint: disable=C0103


class ConvTranspose2d(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        activation=None,
        dtype="float16",
    ):
        """Initialize the ConvTranspose2d class

        Parameters
        ----------
        in_channel : [type]
            [description]
        out_channel : [type]
            [description]
        kernel_size : [type]
            [description]
        stride : [type]
            [description]
        pad : str, optional
            [description], by default 'SAME'
        dilate : int, optional
            [description], by default 1
        dtype : str, optional
            [description], by default "float16"

        Raises
        ------
        NotImplementedError
            [description]
        """
        super().__init__()
        self.weight = Parameter(
            shape=[in_channels, kernel_size, kernel_size, out_channels // groups],
            dtype=dtype,
        )
        if bias:
            self.bias = Parameter(shape=[out_channels], dtype=dtype)
        else:
            self.bias = None
        self.op = ops.transposed_conv2d(
            stride=stride,
            pad=padding,
            dilate=dilation,
            group=groups,
            bias=bias,
            activation=activation,
        )

    def forward(self, x: Tensor):
        return self.op(
            x=x,
            w=self.weight.tensor(),
            b=self.bias.tensor() if self.bias is not None else None,
        )
