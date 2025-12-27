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

import torch
from dinoml.compiler import compile_model

from dinoml.frontend import Tensor
from dinoml.frontend.nn.activation import GELU
from dinoml.testing import detect_target
from dinoml.testing.test_utils import (
    get_random_torch_tensor,
    get_torch_empty_tensor,
)


class GELUTestCase(unittest.TestCase):
    def _test_gelu(self, approximate, dtype="float16"):
        input_shape = (3, 10, 20)

        X_pt = get_random_torch_tensor(input_shape, dtype=dtype)
        OP_pt = torch.nn.GELU(approximate=approximate).cuda().half()
        Y_pt = OP_pt(X_pt)
        X_dinoml = Tensor(
            shape=input_shape,
            dtype=dtype,
            name="input0",
            is_input=True,
        )
        OP_dinoml = GELU(approximate=approximate)
        Y_dinoml = OP_dinoml(X_dinoml)

        Ys_dinoml = Ys_dinoml = [
            var._attrs["values"][0] for var in Y_dinoml._attrs["shape"]
        ]
        self.assertEqual(list(Y_pt.shape), Ys_dinoml)

        Y_dinoml._attrs["name"] = "output_0"
        Y_dinoml._attrs["is_output"] = True

        target = detect_target()
        module = compile_model(Y_dinoml, target, "./tmp", "gelu")

        y = get_torch_empty_tensor(Ys_dinoml, dtype=dtype)
        inputs = {"input0": X_pt}
        module.run_with_tensors(inputs, [y])

        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-2, rtol=1e-2))

    def test_gelu(self):
        self._test_gelu(approximate="none")
        self._test_gelu(approximate="tanh")


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
