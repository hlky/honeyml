import torch

from dinoml.compiler import compile_model, ops
from dinoml.frontend import Tensor
from dinoml.testing import detect_target
from dinoml.testing.benchmark_dinoml import benchmark_module


torch.manual_seed(0)


def _run_case(shape, dims, dtype=torch.float16):
    x_pt = torch.randn(shape, device="cuda", dtype=dtype)
    y_pt = torch.flip(x_pt, dims=dims)

    x = Tensor(list(shape), name="x", is_input=True, dtype=str(dtype).split(".")[-1])
    y = ops.flip(dims)(x)
    y._attrs["name"] = "y"
    y._attrs["is_output"] = True

    module = compile_model(
        y, detect_target(), "./tmp", f"flip_{len(shape)}d_{len(dims)}dims"
    )

    out = module.run_with_tensors({"x": x_pt}, {"y": torch.empty_like(y_pt)})["y"]

    torch.testing.assert_close(out, y_pt, rtol=1e-5, atol=1e-5)
    benchmark_module(module, count=100)


def test_flip_various():
    # 1D
    _run_case((4096,), dims=(0,), dtype=torch.float16)

    # 3D: match the style of other tests
    _run_case((1, 512, 4096), dims=(2,), dtype=torch.bfloat16)
    _run_case((1, 512, 4096), dims=(1, 2), dtype=torch.bfloat16)
    _run_case((1, 512, 4096), dims=(0, 2), dtype=torch.bfloat16)

    # 4D with negative dims
    _run_case((2, 3, 5, 7), dims=(-1,), dtype=torch.float16)
    _run_case((2, 3, 5, 7), dims=(-1, -3), dtype=torch.float16)

    # 5D, multi dims
    _run_case((2, 2, 3, 4, 5), dims=(0, 2, 4), dtype=torch.float16)


test_flip_various()
