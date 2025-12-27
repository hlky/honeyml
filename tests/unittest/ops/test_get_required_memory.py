import unittest

import torch

from dinoml.compiler import compile_model, ops
from dinoml.frontend import Tensor
from dinoml.testing import detect_target


class GetRequiredMemoryTestCase(unittest.TestCase):
    def test_get_required_memory(
        self,
    ):
        target = detect_target()
        X = Tensor(
            shape=[10],
            dtype="float16",
            name="input_0",
            is_input=True,
        )
        OP = ops.pad(pad=(1, 1), mode="constant", value=1.0)
        Y = OP(X)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", "get_required_memory")
        required_memory = module.get_required_memory()
        assert required_memory > 0
        print(required_memory)


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
