import torch

from dinoml.compiler import compile_model, ops
from dinoml.testing import detect_target
from dinoml.testing.benchmark_dinoml import benchmark_module
from dinoml.testing.benchmark_pt import benchmark_torch_function


def kdownsample_weight(
    channels, device=None, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    kernel_1d = torch.tensor([[1 / 8, 3 / 8, 3 / 8, 1 / 8]])
    kernel = kernel_1d.T @ kernel_1d
    weight = torch.zeros(
        [
            channels,
            channels,
            kernel.shape[0],
            kernel.shape[1],
        ],
        device=device,
        dtype=dtype,
    )
    indices = torch.arange(
        channels,
        device=device,
    )
    kernel = kernel.to(weight)[None, :].expand(channels, -1, -1)
    weight[indices, indices] = kernel
    return weight


for dtype, torch_dtype in [("float32", torch.float32), ("float16", torch.float16)]:
    for channels in [384, 768]:

        torch.manual_seed(0)

        y_pt = kdownsample_weight(channels, "cuda", torch_dtype)

        y = ops.kdownsample2d_weight()(channels, dtype)
        y._attrs["name"] = "y"
        y._attrs["is_output"] = True

        module = compile_model(y, detect_target(), "./tmp", f"kdownsample2d_weight_{dtype}_{channels}")

        out = module.run_with_tensors({}, {"y": torch.empty_like(y_pt).contiguous()})[
            "y"
        ]

        torch.testing.assert_close(out, y_pt, rtol=1e-5, atol=1e-5)

        mean, _ = benchmark_module(module, count=100)

        pt_mean = benchmark_torch_function(
            100, kdownsample_weight, channels, "cuda", torch_dtype
        )
        print(
            f"DinoML {dtype} {channels} mean:",
            mean,
            "PT mean:",
            pt_mean,
            "speedup:",
            pt_mean / mean,
        )
