import unittest

import torch

from dinoml.compiler import compile_model, ops
from dinoml.frontend import Tensor
from dinoml.testing import detect_target
from dinoml.testing.test_utils import get_random_torch_tensor


class CreateStreamTestCase(unittest.TestCase):
    def test_create_stream(
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
        module = compile_model(Y, target, "./tmp", "create_stream")
        X_pt = get_random_torch_tensor([10], dtype="float16")
        Y_pt = torch.nn.functional.pad(X_pt, (1, 1), mode="constant", value=1.0)
        x = X_pt
        y = torch.empty_like(Y_pt)
        stream_ptr = module.create_stream()
        module.run_with_tensors([x], [y], stream_ptr=stream_ptr)
        torch.testing.assert_close(
            y,
            Y_pt.to(y.dtype),
            rtol=1e-3,
            atol=1e-3,
            msg=lambda msg: f"{msg}\n\ncreate_stream\npt ({Y_pt.shape}):\n{Y_pt}\n\ndinoml ({y.shape}):\n{y}\n\n",
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
