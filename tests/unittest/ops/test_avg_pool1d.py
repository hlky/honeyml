import unittest

import torch
from honey.compiler import compile_model

from honey.frontend import IntVar, nn, Tensor
from honey.testing import detect_target
from honey.testing.test_utils import get_random_torch_tensor


class AvgPoolTestCase(unittest.TestCase):
    def _test_avg_pool_1d(self, dtype="float16"):
        batch_size = [1, 3]
        target = detect_target()
        X = Tensor(
            shape=[IntVar(values=batch_size, name="input_batch"), 7, 2048],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        OP = nn.AvgPool1d(kernel_size=7, stride=1, padding=0)
        Y = OP(X)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", "avg_pool1d")
        for batch in batch_size:
            X_pt = get_random_torch_tensor([batch, 2048, 7], dtype=dtype)
            OP_pt = torch.nn.AvgPool1d(kernel_size=7, stride=1, padding=0)
            Y_pt = OP_pt(X_pt)
            x = torch.permute(X_pt, (0, 2, 1)).contiguous()
            y = torch.empty_like(Y_pt).permute(0, 2, 1).contiguous()
            module.run_with_tensors([x], [y])
            y_transpose = torch.permute(y, (0, 2, 1))
            self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=1e-2, rtol=1e-2))

    def test_avg_pool_1d_fp16(self):
        self._test_avg_pool_1d(dtype="float16")

    @unittest.skipIf(detect_target().name() == "rocm", "fp32 not supported in ROCm")
    def test_avg_pool_1d_fp32(self):
        self._test_avg_pool_1d(dtype="float32")


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
