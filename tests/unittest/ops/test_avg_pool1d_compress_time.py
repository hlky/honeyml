import unittest

import torch
import torch.nn.functional as F

from dinoml.compiler import compile_model, ops

from dinoml.frontend import IntVar, Tensor
from dinoml.testing import detect_target
from dinoml.testing.test_utils import get_random_torch_tensor


def avg_pool_1d_compress_time_ref(x: torch.Tensor) -> torch.Tensor:
    batch_size, channels, frames, height, width = x.shape

    # (batch_size, channels, frames, height, width) -> (batch_size, height, width, channels, frames) -> (batch_size * height * width, channels, frames)
    x = x.permute(0, 3, 4, 1, 2).reshape(batch_size * height * width, channels, frames)

    if x.shape[-1] % 2 == 1:
        x_first, x_rest = x[..., 0], x[..., 1:]
        if x_rest.shape[-1] > 0:
            # (batch_size * height * width, channels, frames - 1) -> (batch_size * height * width, channels, (frames - 1) // 2)
            x_rest = F.avg_pool1d(x_rest, kernel_size=2, stride=2)

        x = torch.cat([x_first[..., None], x_rest], dim=-1)
        # (batch_size * height * width, channels, (frames // 2) + 1) -> (batch_size, height, width, channels, (frames // 2) + 1) -> (batch_size, channels, (frames // 2) + 1, height, width)
        x = x.reshape(batch_size, height, width, channels, x.shape[-1]).permute(
            0, 3, 4, 1, 2
        )
    else:
        # (batch_size * height * width, channels, frames) -> (batch_size * height * width, channels, frames // 2)
        x = F.avg_pool1d(x, kernel_size=2, stride=2)
        # (batch_size * height * width, channels, frames // 2) -> (batch_size, height, width, channels, frames // 2) -> (batch_size, channels, frames // 2, height, width)
        x = x.reshape(batch_size, height, width, channels, x.shape[-1]).permute(
            0, 3, 4, 1, 2
        )
    return x


def avg_pool_1d_compress_time(x: Tensor) -> Tensor:
    batch_size, frames, height, width, channels = x._attrs["shape"]

    # (batch_size, frames, height, width, channels) -> (batch_size, height, width, channels, frames) -> (batch_size * height * width, channels, frames)
    x = ops.reshape()(ops.permute()(x, [0, 2, 3, 4, 1]), [-1, channels, frames])

    x = ops.avg_pool1d_compress_time()(x)

    # (batch_size * height * width, channels, frames // 2 or (frames - 1) // 2) -> (batch_size, height, width, channels, frames // 2 or (frames - 1) // 2) -> (batch_size, frames // 2 or (frames - 1) // 2, height, width, channels)
    x = ops.permute()(
        ops.reshape()(x, [batch_size, height, width, channels, x._attrs["shape"][-1]]),
        [0, 4, 1, 2, 3],
    )
    return x


class AvgPoolTestCase(unittest.TestCase):
    def _test_avg_pool_1d(self, dtype="float16"):
        batch_size = [1, 3]
        frames = [1, 4, 7]
        target = detect_target()
        X = Tensor(
            shape=[
                IntVar(values=batch_size, name="input_batch"),
                IntVar([1, 8], name="input_frames"),
                16,
                24,
                8,
            ],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        OP = avg_pool_1d_compress_time
        Y = OP(X)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", f"avg_pool1d_compress_time_{dtype}")
        for batch in batch_size:
            for frame in frames:
                X_pt = get_random_torch_tensor([batch, 8, frame, 16, 24], dtype=dtype)
                OP_pt = avg_pool_1d_compress_time_ref
                Y_pt = OP_pt(X_pt)
                x = torch.permute(X_pt, (0, 2, 3, 4, 1)).contiguous()
                y = torch.empty_like(Y_pt).permute(0, 2, 3, 4, 1).contiguous()
                module.run_with_tensors([x], [y])
                y_transpose = torch.permute(y, (0, 4, 1, 2, 3)).contiguous()
                self.assertTrue(torch.allclose(Y_pt, y_transpose, atol=1e-2, rtol=1e-2))

    def test_avg_pool_1d_fp16(self):
        self._test_avg_pool_1d(dtype="float16")

    @unittest.skipIf(detect_target().name() == "rocm", "fp32 not supported in ROCm")
    def test_avg_pool_1d_fp32(self):
        self._test_avg_pool_1d(dtype="float32")


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
