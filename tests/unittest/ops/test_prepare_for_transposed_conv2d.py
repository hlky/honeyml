import torch
from dinoml.compiler import compile_model, ops
from dinoml.frontend import Tensor
from dinoml.testing import detect_target


def test_prepare_for_transposed_conv2d():
    batch, in_channels, h, w = 2, 64, 32, 32
    x = torch.randn(batch, in_channels, h, w, device="cuda")
    stride = (2, 2)

    up_h = (h - 1) * stride[0] + 1
    up_w = (w - 1) * stride[1] + 1

    upsampled = torch.zeros(
        batch, in_channels, up_h, up_w, device=x.device, dtype=x.dtype
    )
    upsampled[:, :, :: stride[0], :: stride[1]] = x

    x_t = Tensor([batch, h, w, in_channels], is_input=True, name="x", dtype="float32")
    y = ops.prepare_for_transposed_conv2d(stride)(x_t)
    y._attrs["name"] = "y"
    y._attrs["is_output"] = True

    mod = compile_model(y, detect_target(), "./tmp", "prep_transpose")
    out = mod.run_with_tensors(
        {"x": x.permute(0, 2, 3, 1).contiguous()},
        {"y": torch.empty_like(upsampled).permute(0, 2, 3, 1).contiguous()},
    )["y"]

    torch.testing.assert_close(out, upsampled.permute(0, 2, 3, 1).contiguous())


test_prepare_for_transposed_conv2d()
