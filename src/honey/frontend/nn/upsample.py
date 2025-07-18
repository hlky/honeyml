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
Unsampling2d module.
"""
from honey.compiler.ops import (
    upsampling1d,
    upsampling1d_add,
    upsampling2d,
    upsampling2d_add,
)
from honey.frontend.nn.module import Module


class Upsampling2d(Module):
    r"""
    Applies a 2D bilinear upsampling to an input signal composed of several input
    channels.

    To specify the scale, it takes the :attr:`scale_factor` as it's constructor argument.

    * :attr:`scale_factor` (float): multiplier for spatial size.

    * :attr:`mode` (str): the upsampling algorithm: one of ``'nearest'``,
      ``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``.
      Currently we support ``'bilinear'`` and  ``'nearest'`` mode.

    Args:
        input (Tensor [N, H, W, C]): the input data.

    Return:
        Tensor [N, H_out, W_out, C].
    """

    def __init__(self, scale_factor, mode, align_corners=False):
        super().__init__()
        self.op = upsampling2d(scale_factor, mode, align_corners)

    def forward(self, *args):
        out = None
        x = args[0]
        if len(args) == 2:
            out = args[1]
        return self.op(x, out)


class Upsampling2dAdd(Module):
    r"""Applies Upsampling2d + add."""

    def __init__(self, scale_factor, mode, align_corners=False):
        super().__init__()
        self.op = upsampling2d_add(scale_factor, mode, align_corners)

    def forward(self, *args):
        assert len(args) == 2
        x = args[0]
        res = args[1]
        return self.op(x, res)


class Upsampling1d(Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super().__init__()
        self.op = upsampling1d(scale_factor, mode, align_corners)

    def forward(self, *args):
        out = None
        x = args[0]
        if len(args) == 2:
            out = args[1]
        return self.op(x, out)


class Upsampling1dAdd(Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super().__init__()
        self.op = upsampling1d_add(scale_factor, mode, align_corners)

    def forward(self, *args):
        assert len(args) == 2
        x = args[0]
        res = args[1]
        return self.op(x, res)
