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
conv2d depthwise module
"""
from honey.compiler.ops import conv2d_depthwise, conv2d_depthwise_bias
from honey.frontend.nn.conv2d.conv2d import Conv2d


class Conv2dDepthwise(Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        dtype="float16",
        bias=False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            dtype,
            bias=bias,
        )
        op_cls = conv2d_depthwise_bias if self.bias_flag else conv2d_depthwise
        self.op = op_cls(stride=stride, pad=padding, dilate=dilation, group=groups)
