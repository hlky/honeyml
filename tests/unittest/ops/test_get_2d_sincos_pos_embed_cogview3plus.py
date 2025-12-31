from typing import Tuple, Union
import torch

from dinoml.compiler import compile_model, ops
from dinoml.testing import detect_target
from dinoml.testing.benchmark_dinoml import benchmark_module
from dinoml.testing.benchmark_pt import benchmark_torch_function


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = torch.arange(embed_dim // 2, device=pos.device, dtype=torch.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = torch.outer(pos, omega)
    return torch.concat([torch.sin(out), torch.cos(out)], dim=1)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # uses grid[0]
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # uses grid[1]
    return torch.concat([emb_h, emb_w], dim=1)


def get_2d_sincos_pos_embed_table_ref(
    embed_dim: int,
    grid_size: Union[int, Tuple[int, int]],
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
    pos_embed = get_2d_sincos_pos_embed_from_grid(
        embed_dim, grid
    )  # [H*W, D] where H=grid_size[0],W=grid_size[1]

    # match the snippet: reshape to [P, P, D] when grid is square
    H, W = grid_size
    pos_embed = pos_embed.reshape(H, W, embed_dim)
    return pos_embed


def cogview3plus_joint_ref(
    embed_dim,
    grid_size,
    interpolation_scale,
    base_size,
    height: int,
    width: int,
    text_length: int,
) -> torch.Tensor:
    pos_table = get_2d_sincos_pos_embed_table_ref(
        embed_dim,
        grid_size,
        interpolation_scale=interpolation_scale,
        base_size=base_size,
        device="cuda",
    )
    image_pos_embed = pos_table[:height, :width].reshape(height * width, -1)
    text_pos_embed = torch.zeros(
        (text_length, pos_table.shape[-1]),
        device=pos_table.device,
        dtype=pos_table.dtype,
    )
    out = torch.cat([text_pos_embed, image_pos_embed], dim=0)[None, ...]
    return out


torch.manual_seed(0)

HIDDEN = 256
POS_MAX = 32
HEIGHT = 24
WIDTH = 16
TEXT_LEN = 77
INTERP = 1.0
BASE = POS_MAX  # matches the snippet call: base_size=pos_embed_max_size


y_pt = cogview3plus_joint_ref(
    HIDDEN, (POS_MAX, POS_MAX), INTERP, BASE, HEIGHT, WIDTH, TEXT_LEN
)

pos_table = ops.get_2d_sincos_pos_embed()(
    embed_dim=HIDDEN,
    grid_size=(POS_MAX, POS_MAX),
    interpolation_scale=INTERP,
    base_size=BASE,
)

y = ops.get_2d_sincos_pos_embed_cogview3plus()(
    pos_table=pos_table,
    hidden_size=HIDDEN,
    pos_embed_max_size=POS_MAX,
    height=HEIGHT,
    width=WIDTH,
    text_length=TEXT_LEN,
)
y._attrs["name"] = "y"
y._attrs["is_output"] = True

module = compile_model(
    y, detect_target(), "./tmp", "get_2d_sincos_pos_embed_cogview3plus"
)

out = module.run_with_tensors({}, {"y": torch.empty_like(y_pt)})["y"]
torch.testing.assert_close(out, y_pt, rtol=1e-5, atol=1e-5)

mean, _ = benchmark_module(module, count=100)

pt_mean = benchmark_torch_function(
    100,
    cogview3plus_joint_ref,
    HIDDEN,
    (POS_MAX, POS_MAX),
    INTERP,
    BASE,
    HEIGHT,
    WIDTH,
    TEXT_LEN,
)
print("DinoML mean:", mean, "ms")
print("PyTorch mean:", pt_mean, "ms")
print(f"DinoML is {pt_mean / mean:.2f}x PyTorch")
