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
import unittest
from collections import OrderedDict

import torch
import torch as pt
from honey import frontend as honey

from honey.compiler import ops


class NNModule(unittest.TestCase):
    def test_module(self):
        class HoneyModule(honey.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = None
                self.b = None
                self.w = honey.Parameter(shape=[32, 32], dtype="float16")
                self.b = honey.Parameter(
                    shape=[
                        32,
                    ],
                    dtype="float16",
                )

            def forward(self, x):
                return ops.gemm_rcr_bias()(x, self.w.tensor(), self.b.tensor())

        class PTModule(pt.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = pt.nn.Parameter(torch.randn(32, 32))
                self.b = pt.nn.Parameter(
                    torch.randn(
                        32,
                    )
                )

            def forward(self, x):
                return pt.mm(x, self.w) + self.b

        a = HoneyModule()
        honey_param_names = [x[0] for x in a.named_parameters()]

        b = PTModule()
        pt_param_names = [x[0] for x in b.named_parameters()]

        for x, y in zip(honey_param_names, pt_param_names):
            assert x == y

    def test_sequential_1(self):
        class HoneyModule(honey.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = honey.Parameter(shape=[32, 32], dtype="float16")
                self.b = honey.Parameter(
                    shape=[
                        32,
                    ],
                    dtype="float16",
                )

            def forward(self, x):
                return ops.gemm_rcr_bias()(x, self.w.tensor(), self.b.tensor())

        class PTModule(pt.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = pt.nn.Parameter(torch.randn(32, 32))
                self.b = pt.nn.Parameter(
                    torch.randn(
                        32,
                    )
                )

            def forward(self, x):
                return pt.mm(x, self.w) + self.b

        a = honey.nn.Sequential(HoneyModule(), HoneyModule(), HoneyModule())
        b = pt.nn.Sequential(
            PTModule(),
            PTModule(),
            PTModule(),
        )

        honey_param_names = [x[0] for x in a.named_parameters()]
        pt_param_names = [x[0] for x in b.named_parameters()]

        for x, y in zip(honey_param_names, pt_param_names):
            assert x == y

    def test_sequential_2(self):
        class HoneyModule(honey.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = honey.Parameter(shape=[32, 32], dtype="float16")
                self.b = honey.Parameter(
                    shape=[
                        32,
                    ],
                    dtype="float16",
                )

            def forward(self, x):
                return ops.gemm_rcr_bias()(x, self.w.tensor(), self.b.tensor())

        class PTModule(pt.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = pt.nn.Parameter(torch.randn(32, 32))
                self.b = pt.nn.Parameter(
                    torch.randn(
                        32,
                    )
                )

            def forward(self, x):
                return pt.mm(x, self.w) + self.b

        a = honey.nn.Sequential(
            OrderedDict(
                [
                    ("block1", HoneyModule()),
                    ("block2", HoneyModule()),
                    ("block3", HoneyModule()),
                ]
            )
        )
        b = pt.nn.Sequential(
            OrderedDict(
                [
                    ("block1", PTModule()),
                    ("block2", PTModule()),
                    ("block3", PTModule()),
                ]
            )
        )

        honey_param_names = [x[0] for x in a.named_parameters()]
        pt_param_names = [x[0] for x in b.named_parameters()]

        for x, y in zip(honey_param_names, pt_param_names):
            assert x == y

    def test_module_dict(self):
        class HoneyModule(honey.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = honey.Parameter(shape=[32, 32], dtype="float16")
                self.b = honey.Parameter(
                    shape=[
                        32,
                    ],
                    dtype="float16",
                )

            def forward(self, x):
                return ops.gemm_rcr_bias()(x, self.w.tensor(), self.b.tensor())

        class PTModule(pt.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = pt.nn.Parameter(torch.randn(32, 32))
                self.b = pt.nn.Parameter(
                    torch.randn(
                        32,
                    )
                )

            def forward(self, x):
                return pt.mm(x, self.w) + self.b

        class HoneyDict(honey.nn.Module):
            def __init__(self):
                super().__init__()
                self.dict_a = honey.nn.ModuleDict(
                    {
                        "block1": HoneyModule(),
                        "block2": HoneyModule(),
                    }
                )
                self.dict_b = honey.nn.ModuleDict(
                    {
                        "block_a": HoneyModule(),
                        "block_b": HoneyModule(),
                    }
                )

            def forward(self, x):
                return (
                    self.dict_a["block1"](x)
                    + self.dict_a["block2"](x)
                    + self.dict_b["block_a"](x)
                    + self.dict_b["block_b"](x)
                )

        class PTDict(pt.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dict_a = pt.nn.ModuleDict(
                    {
                        "block1": PTModule(),
                        "block2": PTModule(),
                    }
                )
                self.dict_b = pt.nn.ModuleDict(
                    {
                        "block_a": PTModule(),
                        "block_b": PTModule(),
                    }
                )

            def forward(self, x):
                return (
                    self.dict_a["block1"](x)
                    + self.dict_a["block2"](x)
                    + self.dict_b["block_a"](x)
                    + self.dict_b["block_b"](x)
                )

        a = HoneyDict()
        b = PTDict()

        honey_param_names = [x[0] for x in a.named_parameters()]
        pt_param_names = [x[0] for x in b.named_parameters()]

        for x, y in zip(honey_param_names, pt_param_names):
            assert x == y


if __name__ == "__main__":
    unittest.main()
