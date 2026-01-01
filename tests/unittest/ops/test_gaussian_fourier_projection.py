import math
import numpy as np
import torch

from dinoml.compiler import compile_model, ops
from dinoml.frontend import Tensor
from dinoml.testing import detect_target
from dinoml.testing.benchmark_dinoml import benchmark_module
from dinoml.testing.benchmark_pt import benchmark_torch_function


def ref_gaussian_fourier_projection(
    x: torch.Tensor, weight: torch.Tensor, log: bool, flip_sin_to_cos: bool
):
    if log:
        x = torch.log(x)

    x_proj = x[:, None] * weight[None, :] * 2 * np.pi

    if flip_sin_to_cos:
        out = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)
    else:
        out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    return out


torch.manual_seed(0)

N = 1024
E = 256

TOLERANCE = {"float32": 3e-5, "float16": 2e-2}

for dtype in ["float32", "float16"]:
    for log in [False, True]:
        for flip in [False, True]:
            # Use positive x because log(x)
            if dtype == "float32":
                torch_dtype = torch.float32
            else:
                torch_dtype = torch.float16
            x_pt = torch.rand([N], device="cuda", dtype=torch_dtype) + 1e-2
            weight_pt = (
                torch.randn([E], device="cuda", dtype=torch_dtype)
            ).contiguous()

            y_ref = ref_gaussian_fourier_projection(x_pt, weight_pt, log, flip)

            x = Tensor([N], name="x", is_input=True, dtype=dtype)
            w = Tensor([E], name="w", is_input=True, dtype=dtype)

            y = ops.gaussian_fourier_projection()(x, w, log, flip)
            y._attrs["name"] = "y"
            y._attrs["is_output"] = True

            module = compile_model(
                y,
                detect_target(),
                "./tmp",
                f"gaussian_fourier_projection_{dtype}_{log}_{flip}",
            )

            out = module.run_with_tensors(
                {"x": x_pt.contiguous(), "w": weight_pt.contiguous()},
                {"y": torch.empty_like(y_ref).contiguous()},
            )["y"]

            if dtype == "float16" and log:
                # TODO: find real model that uses `GaussianFourierProjection` with log=True
                # find real weight values, input, and dtype
                torch.testing.assert_close(out, y_ref, rtol=1e-1, atol=1e-1)
            else:
                torch.testing.assert_close(
                    out, y_ref, rtol=TOLERANCE[dtype], atol=TOLERANCE[dtype]
                )

            mean, _ = benchmark_module(module, count=100)
            pt_mean = benchmark_torch_function(
                100, ref_gaussian_fourier_projection, x_pt, weight_pt, log, flip
            )
            print("DinoML mean:", mean, "PT mean:", pt_mean, "speedup:", pt_mean / mean)
