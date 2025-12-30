import math
import torch

from dinoml.compiler import compile_model, ops
from dinoml.frontend import Tensor
from dinoml.testing import detect_target
from dinoml.testing.benchmark_dinoml import benchmark_module
from dinoml.testing.benchmark_pt import benchmark_torch_function


def get_timestep_embedding_ref(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


torch.manual_seed(0)

# Test a couple of common settings (including odd embedding_dim and flip)
test_cases = [
    dict(
        embedding_dim=320,
        flip_sin_to_cos=False,
        downscale_freq_shift=1.0,
        scale=1.0,
        max_period=10000,
    ),
    dict(
        embedding_dim=321,
        flip_sin_to_cos=True,
        downscale_freq_shift=1.0,
        scale=1.0,
        max_period=10000,
    ),
    dict(
        embedding_dim=128,
        flip_sin_to_cos=False,
        downscale_freq_shift=0.0,
        scale=2.0,
        max_period=10000,
    ),
]

for cfg in test_cases:
    N = 2
    timesteps_pt = (
        torch.linspace(1, 999, steps=N, device="cuda", dtype=torch.float32) / 1000
    )
    print(timesteps_pt)

    y_pt = get_timestep_embedding_ref(
        timesteps_pt,
        **cfg,
    )

    timesteps = Tensor([N], name="timesteps", is_input=True, dtype="float32")
    y = ops.get_timestep_embedding()(
        timesteps,
        embedding_dim=cfg["embedding_dim"],
        flip_sin_to_cos=cfg["flip_sin_to_cos"],
        downscale_freq_shift=cfg["downscale_freq_shift"],
        scale=cfg["scale"],
        max_period=cfg["max_period"],
    )
    y._attrs["name"] = "y"
    y._attrs["is_output"] = True

    module = compile_model(
        y, detect_target(), "./tmp", f"get_timestep_embedding_{cfg['embedding_dim']}"
    )

    out = module.run_with_tensors(
        {"timesteps": timesteps_pt},
        {"y": torch.empty_like(y_pt)},
    )["y"]

    diff = (y_pt - out).abs()
    print("max diff:", diff.max().item())
    print("mean diff:", diff.mean().item())

    torch.testing.assert_close(out, y_pt, rtol=3e-4, atol=3e-4)

    mean, _ = benchmark_module(module, count=100)

    pt_mean = benchmark_torch_function(
        100, get_timestep_embedding_ref, timesteps_pt, **cfg
    )
    print("DinoML mean:", mean, "ms")
    print("PyTorch mean:", pt_mean, "ms")
    print(f"DinoML is {pt_mean / mean:.2f}x PyTorch")
