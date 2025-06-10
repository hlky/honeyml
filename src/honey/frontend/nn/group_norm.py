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
GroupNorm module
"""
from honey.compiler import ops
from honey.frontend.nn.module import Module
from honey.frontend.nn.parameter import Parameter

# pylint: disable=C0103


class GroupNorm(Module):
    """GroupNorm nn module"""

    def __init__(
        self,
        num_groups,
        num_channels,
        eps=1e-5,
        affine=True,
        dtype="float16",
        use_swish=False,
        **kwargs,
    ):
        """Group Norm init"""
        super().__init__()
        self.dtype = dtype
        self.eps = eps
        op_name = "group_norm_swish" if use_swish else "group_norm"
        self.weight = Parameter(shape=[num_channels], dtype=dtype)
        self.bias = Parameter(shape=[num_channels], dtype=dtype)
        self.op = getattr(ops, op_name)(num_groups, num_channels)

    def forward(self, *args):
        assert len(args) == 1
        x = args[0]
        dtype = x.dtype()
        y = self.op(
            x,
            (
                self.weight.tensor()
                if dtype == self.dtype
                else ops.cast()(self.weight.tensor(), dtype)
            ),
            (
                self.bias.tensor()
                if dtype == self.dtype
                else ops.cast()(self.bias.tensor(), dtype)
            ),
            self.eps,
        )
        return y
