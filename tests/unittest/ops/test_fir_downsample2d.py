import torch
import torch.nn.functional as F

from diffusers.models.downsampling import FirDownsample2D

from dinoml.compiler import compile_model
from dinoml.frontend import Tensor
from dinoml.testing import detect_target
from dinoml.compiler import ops
from dinoml.testing.benchmark_dinoml import benchmark_module
from dinoml.testing.benchmark_pt import benchmark_torch_function


def test_fir_downsample2d():
    torch.manual_seed(0)
    device = "cuda"
    channels = 64
    x_pt = torch.randn([2, channels, 64, 64], device=device, dtype=torch.float32)

    ref = FirDownsample2D(
        channels=channels,
        use_conv=False,
    ).to(device, torch.float32).eval()

    with torch.no_grad():
        y_ref = ref(x_pt)

    x_nhwc = x_pt.permute(0, 2, 3, 1).contiguous()

    x = Tensor(
        [x_nhwc.shape[0], x_nhwc.shape[1], x_nhwc.shape[2], x_nhwc.shape[3]],
        name="x",
        is_input=True,
        dtype="float32",
    )

    y = ops.fir_downsample2d()(x)
    y._attrs["name"] = "y"
    y._attrs["is_output"] = True

    module = compile_model(y, detect_target(), "./tmp", "fir_downsample2d")

    y_out = module.run_with_tensors(
        {"x": x_nhwc},
        {"y": torch.empty_like(y_ref).contiguous()},
    )["y"]

    y_out_nchw = y_out.permute(0, 3, 1, 2).contiguous()

    torch.testing.assert_close(
        y_out_nchw,
        y_ref,
        rtol=1e-5,
        atol=1e-5,
    )

    mean, _ = benchmark_module(module, count=100)
    pt_mean = benchmark_torch_function(100, ref, x_pt)
    print(
        "DinoML test_fir_downsample2d mean:",
        mean,
        "PT mean:",
        pt_mean,
        "speedup:",
        pt_mean / mean,
    )


def test_fir_downsample2d_with_conv():
    torch.manual_seed(0)
    device = "cuda"

    channels = 64
    out_channels = 96

    x_pt = torch.randn([2, channels, 64, 64], device=device, dtype=torch.float32)

    ref = (
        FirDownsample2D(
            channels=channels,
            out_channels=out_channels,
            use_conv=True,
        )
        .to(device, torch.float32)
        .eval()
    )

    with torch.no_grad():
        y_ref = ref(x_pt)

    x_nhwc = x_pt.permute(0, 2, 3, 1).contiguous()

    w = ref.Conv2d_0.weight.detach()
    b = ref.Conv2d_0.bias.detach()

    w_ohwi = w.permute(0, 2, 3, 1).contiguous()

    x = Tensor([*x_nhwc.shape], name="x", is_input=True, dtype="float32")
    w_t = Tensor(list(w_ohwi.shape), name="w", is_input=True, dtype="float32")
    b_t = Tensor(list(b.shape), name="b", is_input=True, dtype="float32")

    u = ops.fir_filter_pad2()(x)
    y = ops.conv2d(stride=2, bias=False)(u, w_t)
    b_reshaped = ops.reshape()(b_t, [1, 1, 1, out_channels])
    y = y + b_reshaped
    y._attrs["name"] = "y"
    y._attrs["is_output"] = True

    module = compile_model(y, detect_target(), "./tmp", "fir_downsample2d_conv")

    y_out = module.run_with_tensors(
        {"x": x_nhwc, "w": w_ohwi, "b": b},
        {"y": torch.empty_like(y_ref)},
    )["y"]

    y_out_nchw = y_out.permute(0, 3, 1, 2).contiguous()

    torch.testing.assert_close(
        y_out_nchw,
        y_ref,
        rtol=1e-5,
        atol=1e-5,
    )

    mean, _ = benchmark_module(module, count=100)
    pt_mean = benchmark_torch_function(100, ref, x_pt)
    print(
        "DinoML test_fir_downsample2d_with_conv mean:",
        mean,
        "PT mean:",
        pt_mean,
        "speedup:",
        pt_mean / mean,
    )


test_fir_downsample2d()
test_fir_downsample2d_with_conv()
