from typing import Tuple, Optional
import torch

from dinoml.compiler import compile_model, ops
from dinoml.testing import detect_target
from dinoml.testing.benchmark_dinoml import benchmark_module
from dinoml.testing.benchmark_pt import benchmark_torch_function


def get_1d_rotary_pos_embed(
    dim: int,
    pos: torch.Tensor,
    theta: float = 10000.0,
    linear_factor=1.0,
    ntk_factor=1.0,
    freqs_dtype=torch.float32,  #  torch.float32, torch.float64 (flux)
):
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)

    theta = theta * ntk_factor
    freqs = (
        1.0
        / (
            theta
            ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim)
        )
        / linear_factor
    )  # [D/2]
    freqs = torch.outer(pos, freqs)
    # stable audio, allegro
    freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()  # [S, D]
    freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()  # [S, D]
    return freqs_cos, freqs_sin


def _prepare_rotary_positional_embeddings(
    embed_dim,
    crops_coords,
    grid_size,
    temporal_size,
    interpolation_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    theta: int = 10000,
    device: Optional[torch.device] = None,
):
    start, stop = crops_coords
    grid_size_h, grid_size_w = grid_size
    interpolation_scale_t, interpolation_scale_h, interpolation_scale_w = (
        interpolation_scale
    )
    grid_t = torch.linspace(
        0,
        temporal_size * (temporal_size - 1) / temporal_size,
        temporal_size,
        device=device,
        dtype=torch.float32,
    )
    grid_h = torch.linspace(
        start[0],
        stop[0] * (grid_size_h - 1) / grid_size_h,
        grid_size_h,
        device=device,
        dtype=torch.float32,
    )
    grid_w = torch.linspace(
        start[1],
        stop[1] * (grid_size_w - 1) / grid_size_w,
        grid_size_w,
        device=device,
        dtype=torch.float32,
    )

    # Compute dimensions for each axis
    dim_t = embed_dim // 3
    dim_h = embed_dim // 3
    dim_w = embed_dim // 3

    # Temporal frequencies
    freqs_t = get_1d_rotary_pos_embed(
        dim_t,
        grid_t / interpolation_scale_t,
        theta=theta,
    )
    # Spatial frequencies for height and width
    freqs_h = get_1d_rotary_pos_embed(
        dim_h,
        grid_h / interpolation_scale_h,
        theta=theta,
    )
    freqs_w = get_1d_rotary_pos_embed(
        dim_w,
        grid_w / interpolation_scale_w,
        theta=theta,
    )

    return freqs_t, freqs_h, freqs_w, grid_t, grid_h, grid_w


def get_3d_rotary_pos_embed_allegro(
    height: int,
    width: int,
    num_frames: int,
    device: torch.device,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    interpolation_scale_h: float = 2.0,
    interpolation_scale_t: float = 2.2,
    interpolation_scale_w: float = 2.0,
    attention_head_dim: int = 96,
):
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)

    start, stop = (0, 0), (grid_height, grid_width)
    freqs_t, freqs_h, freqs_w, grid_t, grid_h, grid_w = (
        _prepare_rotary_positional_embeddings(
            embed_dim=attention_head_dim,
            crops_coords=(start, stop),
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
            interpolation_scale=(
                interpolation_scale_t,
                interpolation_scale_h,
                interpolation_scale_w,
            ),
            device=device,
        )
    )

    grid_t = grid_t.to(dtype=torch.long)
    grid_h = grid_h.to(dtype=torch.long)
    grid_w = grid_w.to(dtype=torch.long)

    pos = torch.cartesian_prod(grid_t, grid_h, grid_w)
    pos = pos.reshape(-1, 3).transpose(0, 1).reshape(3, 1, -1).contiguous()
    grid_t, grid_h, grid_w = pos

    return (freqs_t, freqs_h, freqs_w), (grid_t, grid_h, grid_w)


torch.manual_seed(0)

HEIGHT = 256
WIDTH = 320
NUM_FRAMES = 6
VAE = 8
PATCH = 2
S_H = 2.0
S_T = 2.2
S_W = 2.0
HEAD_DIM = 96  # divisible by 3 and (//3) even


(freqs_pt, grids_pt) = get_3d_rotary_pos_embed_allegro(
    HEIGHT,
    WIDTH,
    NUM_FRAMES,
    "cuda",
    VAE,
    PATCH,
    S_H,
    S_T,
    S_W,
    HEAD_DIM,
)

freqs_t_pt, freqs_h_pt, freqs_w_pt = freqs_pt
f_t_cos, f_t_sin = freqs_t_pt
f_h_cos, f_h_sin = freqs_h_pt
f_w_cos, f_w_sin = freqs_w_pt
gt_ref, gh_ref, gw_ref = grids_pt


# DinoML
(freqs, grids) = ops.get_3d_rotary_pos_embed_allegro()(
    height=HEIGHT,
    width=WIDTH,
    num_frames=NUM_FRAMES,
    vae_scale_factor_spatial=VAE,
    patch_size=PATCH,
    interpolation_scale_h=S_H,
    interpolation_scale_t=S_T,
    interpolation_scale_w=S_W,
    attention_head_dim=HEAD_DIM,
)
freqs_t, freqs_h, freqs_w = freqs
t_cos, t_sin = freqs_t
h_cos, h_sin = freqs_h
w_cos, w_sin = freqs_w
grid_t, grid_h, grid_w = grids

# name outputs
t_cos._attrs["name"] = "t_cos"
t_cos._attrs["is_output"] = True
t_sin._attrs["name"] = "t_sin"
t_sin._attrs["is_output"] = True
h_cos._attrs["name"] = "h_cos"
h_cos._attrs["is_output"] = True
h_sin._attrs["name"] = "h_sin"
h_sin._attrs["is_output"] = True
w_cos._attrs["name"] = "w_cos"
w_cos._attrs["is_output"] = True
w_sin._attrs["name"] = "w_sin"
w_sin._attrs["is_output"] = True
grid_t._attrs["name"] = "grid_t"
grid_t._attrs["is_output"] = True
grid_h._attrs["name"] = "grid_h"
grid_h._attrs["is_output"] = True
grid_w._attrs["name"] = "grid_w"
grid_w._attrs["is_output"] = True

module = compile_model(
    [t_cos, t_sin, h_cos, h_sin, w_cos, w_sin, grid_t, grid_h, grid_w],
    detect_target(),
    "./tmp",
    "get_3d_rotary_pos_embed_allegro",
)

outs = module.run_with_tensors(
    {},
    {
        "t_cos": torch.empty_like(f_t_cos).contiguous(),
        "t_sin": torch.empty_like(f_t_sin).contiguous(),
        "h_cos": torch.empty_like(f_h_cos).contiguous(),
        "h_sin": torch.empty_like(f_h_sin).contiguous(),
        "w_cos": torch.empty_like(f_w_cos).contiguous(),
        "w_sin": torch.empty_like(f_w_sin).contiguous(),
        "grid_t": torch.empty_like(gt_ref).contiguous(),
        "grid_h": torch.empty_like(gh_ref).contiguous(),
        "grid_w": torch.empty_like(gw_ref).contiguous(),
    },
)

torch.testing.assert_close(outs["t_cos"], f_t_cos, rtol=1e-5, atol=1e-5)
torch.testing.assert_close(outs["t_sin"], f_t_sin, rtol=1e-5, atol=1e-5)
torch.testing.assert_close(outs["h_cos"], f_h_cos, rtol=1e-5, atol=1e-5)
torch.testing.assert_close(outs["h_sin"], f_h_sin, rtol=1e-5, atol=1e-5)
torch.testing.assert_close(outs["w_cos"], f_w_cos, rtol=1e-5, atol=1e-5)
torch.testing.assert_close(outs["w_sin"], f_w_sin, rtol=1e-5, atol=1e-5)

torch.testing.assert_close(outs["grid_t"], gt_ref)
torch.testing.assert_close(outs["grid_h"], gh_ref)
torch.testing.assert_close(outs["grid_w"], gw_ref)

mean, _ = benchmark_module(module, count=100)
pt_mean = benchmark_torch_function(
    100,
    get_3d_rotary_pos_embed_allegro,
    HEIGHT,
    WIDTH,
    NUM_FRAMES,
    "cuda",
    VAE,
    PATCH,
    S_H,
    S_T,
    S_W,
    HEAD_DIM,
)
print("DinoML mean:", mean, "ms")
print("PyTorch mean:", pt_mean, "ms")
print(f"DinoML is {pt_mean / mean:.2f}x PyTorch")
