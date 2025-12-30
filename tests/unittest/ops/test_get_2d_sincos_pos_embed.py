from typing import Tuple, Union
import torch

from dinoml.compiler import compile_model, ops
from dinoml.testing import detect_target
from dinoml.testing.benchmark_dinoml import benchmark_module
from dinoml.testing.benchmark_pt import benchmark_torch_function


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos, flip_sin_to_cos=False):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = torch.arange(embed_dim // 2, device=pos.device, dtype=torch.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = torch.outer(pos, omega)
    emb = torch.concat([torch.sin(out), torch.cos(out)], dim=1)

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, embed_dim // 2 :], emb[:, : embed_dim // 2]], dim=1)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # uses grid[0]
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # uses grid[1]
    return torch.concat([emb_h, emb_w], dim=1)


def get_2d_sincos_pos_embed_ref(
    embed_dim: int,
    grid_size: Union[int, Tuple[int, int]],
    cls_token: bool = False,
    extra_tokens: int = 0,
    interpolation_scale: float = 1.0,
    base_size: int = 16,
    device: torch.device = "cuda",
) -> torch.Tensor:
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    grid_h = (
        torch.arange(grid_size[0], device=device, dtype=torch.float32)
        / (grid_size[0] / base_size)
        / interpolation_scale
    )
    grid_w = (
        torch.arange(grid_size[1], device=device, dtype=torch.float32)
        / (grid_size[1] / base_size)
        / interpolation_scale
    )
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")  # w first
    grid = torch.stack(grid, dim=0)
    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = torch.concat(
            [torch.zeros([extra_tokens, embed_dim], device=device), pos_embed], dim=0
        )
    return pos_embed


torch.manual_seed(0)

EMBED_DIM = 256
GRID_SIZE = (24, 32)  # (H, W) per reference code usage
INTERP = 1.0
BASE = 16

# test without cls token
y_pt = get_2d_sincos_pos_embed_ref(
    EMBED_DIM,
    GRID_SIZE,
    cls_token=False,
    extra_tokens=0,
    interpolation_scale=INTERP,
    base_size=BASE,
)

y = ops.get_2d_sincos_pos_embed()(
    embed_dim=EMBED_DIM,
    grid_size=GRID_SIZE,
    cls_token=False,
    extra_tokens=0,
    interpolation_scale=INTERP,
    base_size=BASE,
    dtype="float32",
)
y._attrs["name"] = "y"
y._attrs["is_output"] = True

module = compile_model(y, detect_target(), "./tmp", "get_2d_sincos_pos_embed")

out = module.run_with_tensors({}, {"y": torch.empty_like(y_pt)})["y"]
torch.testing.assert_close(out, y_pt, rtol=1e-5, atol=1e-5)

# test with cls token + extra tokens
EXTRA = 2
y_pt2 = get_2d_sincos_pos_embed_ref(
    EMBED_DIM,
    GRID_SIZE,
    cls_token=True,
    extra_tokens=EXTRA,
    interpolation_scale=INTERP,
    base_size=BASE,
)

y2 = ops.get_2d_sincos_pos_embed()(
    embed_dim=EMBED_DIM,
    grid_size=GRID_SIZE,
    cls_token=True,
    extra_tokens=EXTRA,
    interpolation_scale=INTERP,
    base_size=BASE,
    dtype="float32",
)
y2._attrs["name"] = "y"
y2._attrs["is_output"] = True

module2 = compile_model(y2, detect_target(), "./tmp", "get_2d_sincos_pos_embed_cls")

out2 = module2.run_with_tensors({}, {"y": torch.empty_like(y_pt2)})["y"]
torch.testing.assert_close(out2, y_pt2, rtol=1e-5, atol=1e-5)

# benchmark (no-cls case)
mean, _ = benchmark_module(module, count=100)
pt_mean = benchmark_torch_function(
    100, get_2d_sincos_pos_embed_ref, EMBED_DIM, GRID_SIZE, False, 0, INTERP, BASE
)
print("DinoML mean:", mean, "ms")
print("PyTorch mean:", pt_mean, "ms")
print(f"DinoML is {pt_mean / mean:.2f}x PyTorch")
