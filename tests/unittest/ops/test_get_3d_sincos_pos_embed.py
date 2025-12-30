from typing import Tuple, Union
import math
import torch

from dinoml.compiler import compile_model, ops
from dinoml.frontend import Tensor, IntVar
from dinoml.testing import detect_target
from dinoml.testing.benchmark_dinoml import benchmark_module
from dinoml.testing.benchmark_pt import benchmark_torch_function
from dinoml.utils.shape_utils import get_shape


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos, flip_sin_to_cos=False):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = torch.arange(embed_dim // 2, device=pos.device, dtype=torch.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = torch.outer(pos, omega)

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    emb = torch.concat([emb_sin, emb_cos], dim=1)

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, embed_dim // 2 :], emb[:, : embed_dim // 2]], dim=1)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return torch.concat([emb_h, emb_w], dim=1)


def get_3d_sincos_pos_embed_ref(
    embed_dim: int,
    spatial_size: Union[int, Tuple[int, int]],
    temporal_size: int,
    spatial_interpolation_scale: float = 1.0,
    temporal_interpolation_scale: float = 1.0,
) -> torch.Tensor:
    if embed_dim % 4 != 0:
        raise ValueError("`embed_dim` must be divisible by 4")
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)

    embed_dim_spatial = 3 * embed_dim // 4
    embed_dim_temporal = embed_dim // 4

    grid_h = (
        torch.arange(spatial_size[1], dtype=torch.float32, device="cuda")
        / spatial_interpolation_scale
    )
    grid_w = (
        torch.arange(spatial_size[0], dtype=torch.float32, device="cuda")
        / spatial_interpolation_scale
    )
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack(grid, dim=0)
    grid = grid.reshape([2, 1, spatial_size[1], spatial_size[0]])
    pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(embed_dim_spatial, grid)

    grid_t = (
        torch.arange(temporal_size, dtype=torch.float32, device="cuda")
        / temporal_interpolation_scale
    )
    pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(embed_dim_temporal, grid_t)

    pos_embed_spatial = pos_embed_spatial[None, :, :].repeat_interleave(
        temporal_size, dim=0, output_size=temporal_size
    )
    pos_embed_temporal = pos_embed_temporal[:, None, :].repeat_interleave(
        spatial_size[0] * spatial_size[1], dim=1
    )
    return torch.concat([pos_embed_temporal, pos_embed_spatial], dim=-1)


torch.manual_seed(0)

# Choose sizes that satisfy embed_dim % 4 == 0
EMBED_DIM = 256
SPATIAL = (32, 24)  # (W, H) per reference code usage
TEMPORAL = 8
S_SCALE = 1.0
T_SCALE = 1.0

y_pt = get_3d_sincos_pos_embed_ref(EMBED_DIM, SPATIAL, TEMPORAL, S_SCALE, T_SCALE)

print("static")

# DinoML graph (no tensor inputs)
y = ops.get_3d_sincos_pos_embed()(
    embed_dim=EMBED_DIM,
    spatial_size=SPATIAL,
    temporal_size=TEMPORAL,
    spatial_interpolation_scale=S_SCALE,
    temporal_interpolation_scale=T_SCALE,
    dtype="float32",
)
y._attrs["name"] = "y"
y._attrs["is_output"] = True
print(get_shape(y))
module = compile_model(y, detect_target(), "./tmp", "get_3d_sincos_pos_embed")

out = module.run_with_tensors({}, {"y": torch.empty_like(y_pt)})["y"]

diff = (y_pt - out).abs()
print("max diff:", diff.max().item())
print("mean diff:", diff.mean().item())

torch.testing.assert_close(out, y_pt)

mean, _ = benchmark_module(module, count=100)

pt_mean = benchmark_torch_function(
    100, get_3d_sincos_pos_embed_ref, EMBED_DIM, SPATIAL, TEMPORAL, S_SCALE, T_SCALE
)
print(pt_mean)

perf = 1 / (mean / pt_mean)

print(f"DinoML is {perf:.2f}x PyTorch")


## dynamic, no benchmark because requires input tensor and elementwise

print("dynamic")

x = Tensor(
    [IntVar([4, 8]), IntVar([12, 24]), IntVar([16, 32]), EMBED_DIM],
    dtype="float32",
    is_input=True,
    name="x",
)

# DinoML graph (no tensor inputs)
y = ops.get_3d_sincos_pos_embed()(
    embed_dim=EMBED_DIM,
    spatial_size=(x.shape()[2], x.shape()[1]),
    temporal_size=x.shape()[0],
    spatial_interpolation_scale=S_SCALE,
    temporal_interpolation_scale=T_SCALE,
    dtype="float32",
)
# NOTE: use -1 instead of `x.shape()[1] * x.shape()[2]` to avoid codegen issues
x = ops.reshape()(x, (x.shape()[0], -1, x.shape()[3]))
y = y + x
y._attrs["name"] = "y"
y._attrs["is_output"] = True

module = compile_model(y, detect_target(), "./tmp", "get_3d_sincos_pos_embed_dynamic")

out = module.run_with_tensors(
    {"x": torch.zeros([8, 24, 32, 256]).to("cuda", torch.float32).contiguous()},
    {"y": torch.empty_like(y_pt)},
)["y"]

diff = (y_pt - out).abs()
print("max diff:", diff.max().item())
print("mean diff:", diff.mean().item())

torch.testing.assert_close(out, y_pt)
