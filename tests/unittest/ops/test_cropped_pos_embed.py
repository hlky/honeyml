import torch

from dinoml.compiler import compile_model, ops
from dinoml.testing import detect_target
from dinoml.testing.benchmark_dinoml import benchmark_module
from dinoml.testing.benchmark_pt import benchmark_torch_function


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    omega = torch.arange(embed_dim // 2, device=pos.device, dtype=torch.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    out = torch.outer(pos.reshape(-1), omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return torch.cat([emb_h, emb_w], dim=1)


def get_2d_sincos_pos_embed_ref(
    embed_dim,
    grid_size,
    interpolation_scale=1.0,
    base_size=16,
    device="cuda",
):
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
    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


def cropped_pos_embed_ref(
    embed_dim,
    pos_embed_max_size,
    base_size,
    interpolation_scale,
    patch_size,
    height,
    width,
):
    pos_embed = get_2d_sincos_pos_embed_ref(
        embed_dim,
        pos_embed_max_size,
        base_size=base_size,
        interpolation_scale=interpolation_scale,
    )
    height = height // patch_size
    width = width // patch_size

    top = (pos_embed_max_size - height) // 2
    left = (pos_embed_max_size - width) // 2

    spatial_pos_embed = pos_embed.reshape(1, pos_embed_max_size, pos_embed_max_size, -1)
    spatial_pos_embed = spatial_pos_embed[:, top : top + height, left : left + width, :]
    spatial_pos_embed = spatial_pos_embed.reshape(1, -1, spatial_pos_embed.shape[-1])
    return spatial_pos_embed


torch.manual_seed(0)

EMBED_DIM = 1152
POS_MAX = 96
BASE = 64
INTERP = 1.0
PATCH = 2
HEIGHT = 128
WIDTH = 128

y_pt = cropped_pos_embed_ref(
    embed_dim=EMBED_DIM,
    pos_embed_max_size=POS_MAX,
    base_size=BASE,
    interpolation_scale=INTERP,
    patch_size=PATCH,
    height=HEIGHT,
    width=WIDTH,
)

y = ops.cropped_pos_embed()(
    embed_dim=EMBED_DIM,
    pos_embed_max_size=POS_MAX,
    base_size=BASE,
    interpolation_scale=INTERP,
    patch_size=PATCH,
    height=HEIGHT,
    width=WIDTH,
    dtype="float32",
)
y._attrs["name"] = "y"
y._attrs["is_output"] = True

module = compile_model(y, detect_target(), "./tmp", "cropped_pos_embed")

out = module.run_with_tensors({}, {"y": torch.empty_like(y_pt)})["y"]

diff = (y_pt - out).abs()
print("max diff:", diff.max().item())
print("mean diff:", diff.mean().item())

torch.testing.assert_close(out, y_pt, rtol=1e-5, atol=1e-5)

mean, _ = benchmark_module(module, count=200)
pt_mean = benchmark_torch_function(
    200,
    cropped_pos_embed_ref,
    EMBED_DIM,
    POS_MAX,
    BASE,
    INTERP,
    PATCH,
    HEIGHT,
    WIDTH,
)
print("DinoML mean:", mean, "ms")
print("PyTorch mean:", pt_mean, "ms")
print(f"DinoML is {pt_mean / mean:.2f}x PyTorch")
