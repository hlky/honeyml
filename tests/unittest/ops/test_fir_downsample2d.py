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
    x_pt = torch.randn([2, channels, 64, 64], device=device, dtype=torch.float16)

    ref = FirDownsample2D(
        channels=channels,
        use_conv=False,
    ).to(device, torch.float16).eval()

    with torch.no_grad():
        y_ref = ref(x_pt)

    x_nhwc = x_pt.permute(0, 2, 3, 1).contiguous()

    x = Tensor(
        [x_nhwc.shape[0], x_nhwc.shape[1], x_nhwc.shape[2], x_nhwc.shape[3]],
        name="x",
        is_input=True,
        dtype="float16",
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
        rtol=1e-3,
        atol=1e-3,
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

    x_pt = torch.randn([2, channels, 64, 64], device=device, dtype=torch.float16)

    ref = (
        FirDownsample2D(
            channels=channels,
            out_channels=out_channels,
            use_conv=True,
        )
        .to(device, torch.float16)
        .eval()
    )

    with torch.no_grad():
        y_ref = ref(x_pt)

    x_nhwc = x_pt.permute(0, 2, 3, 1).contiguous()

    w = ref.conv.weight.detach()
    b = ref.conv.bias.detach()

    w_hwio = w.permute(2, 3, 1, 0).contiguous()

    x = Tensor([*x_nhwc.shape], name="x", is_input=True, dtype="float16")
    w_t = Tensor(list(w_hwio.shape), name="w", is_input=True, dtype="float16")
    b_t = Tensor(list(b.shape), name="b", is_input=True, dtype="float16")

    y = ops.fir_downsample2d_conv()(x, w_t, b_t)
    y._attrs["name"] = "y"
    y._attrs["is_output"] = True

    module = compile_model(y, detect_target(), "./tmp", "fir_downsample2d_conv")

    y_out = module.run_with_tensors(
        {"x": x_nhwc, "w": w_hwio, "b": b},
        {"y": torch.empty_like(y_ref)},
    )["y"]

    y_out_nchw = y_out.permute(0, 3, 1, 2).contiguous()

    torch.testing.assert_close(
        y_out_nchw,
        y_ref,
        rtol=2e-3,
        atol=2e-3,
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
