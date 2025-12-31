from typing import Optional, Tuple
import torch

from dinoml.compiler import compile_model, ops
from dinoml.testing import detect_target
from dinoml.testing.benchmark_dinoml import benchmark_module
from dinoml.testing.benchmark_pt import benchmark_torch_function


def get_2d_rotary_pos_embed_ref(
    embed_dim,
    crops_coords,
    grid_size,
    use_real=True,
    device: Optional[torch.device] = None,
):
    """
    RoPE for image tokens with 2d structure.

    Args:
    embed_dim: (`int`):
        The embedding dimension size
    crops_coords (`Tuple[int]`)
        The top-left and bottom-right coordinates of the crop.
    grid_size (`Tuple[int]`):
        The grid size of the positional embedding.
    use_real (`bool`):
        If True, return real part and imaginary part separately. Otherwise, return complex numbers.
    device: (`torch.device`, **optional**):
        The device used to create tensors.

    Returns:
        `torch.Tensor`: positional embedding with shape `( grid_size * grid_size, embed_dim/2)`.
    """
    start, stop = crops_coords
    # scale end by (steps−1)/steps matches np.linspace(..., endpoint=False)
    grid_h = torch.linspace(
        start[0],
        stop[0] * (grid_size[0] - 1) / grid_size[0],
        grid_size[0],
        device=device,
        dtype=torch.float32,
    )
    grid_w = torch.linspace(
        start[1],
        stop[1] * (grid_size[1] - 1) / grid_size[1],
        grid_size[1],
        device=device,
        dtype=torch.float32,
    )
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack(grid, dim=0)  # [2, W, H]

    grid = grid.reshape([2, 1, *grid.shape[1:]])
    pos_embed = get_2d_rotary_pos_embed_from_grid(embed_dim, grid, use_real=use_real)
    return pos_embed


def get_2d_rotary_pos_embed_from_grid(embed_dim, grid, use_real=False):
    """
    Get 2D RoPE from grid.

    Args:
    embed_dim: (`int`):
        The embedding dimension size, corresponding to hidden_size_head.
    grid (`np.ndarray`):
        The grid of the positional embedding.
    use_real (`bool`):
        If True, return real part and imaginary part separately. Otherwise, return complex numbers.

    Returns:
        `torch.Tensor`: positional embedding with shape `( grid_size * grid_size, embed_dim/2)`.
    """
    assert embed_dim % 4 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_rotary_pos_embed(
        embed_dim // 2, grid[0].reshape(-1), use_real=use_real
    )  # (H*W, D/2) if use_real else (H*W, D/4)
    emb_w = get_1d_rotary_pos_embed(
        embed_dim // 2, grid[1].reshape(-1), use_real=use_real
    )  # (H*W, D/2) if use_real else (H*W, D/4)

    if use_real:
        cos = torch.cat([emb_h[0], emb_w[0]], dim=1)  # (H*W, D)
        sin = torch.cat([emb_h[1], emb_w[1]], dim=1)  # (H*W, D)
        return cos, sin
    else:
        emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D/2)
        return emb


def get_1d_rotary_pos_embed(
    dim: int,
    pos: torch.Tensor,
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype=torch.float32,  #  torch.float32, torch.float64 (flux)
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        linear_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the context extrapolation. Defaults to 1.0.
        ntk_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the NTK-Aware RoPE. Defaults to 1.0.
        repeat_interleave_real (`bool`, *optional*, defaults to `True`):
            If `True` and `use_real`, real part and imaginary part are each interleaved with themselves to reach `dim`.
            Otherwise, they are concateanted with themselves.
        freqs_dtype (`torch.float32` or `torch.float64`, *optional*, defaults to `torch.float32`):
            the dtype of the frequency tensor.
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    theta = theta * ntk_factor
    freqs = (
        1.0
        / (
            theta
            ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim)
        )
        / linear_factor
    )  # [D/2]
    freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]
    if use_real and repeat_interleave_real:
        # flux, hunyuan-dit, cogvideox
        freqs_cos = (
            freqs.cos()
            .repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2)
            .float()
        )  # [S, D]
        freqs_sin = (
            freqs.sin()
            .repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2)
            .float()
        )  # [S, D]
        return freqs_cos, freqs_sin
    elif use_real:
        # stable audio, allegro
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()  # [S, D]
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        # lumina
        freqs_cis = torch.polar(
            torch.ones_like(freqs), freqs
        )  # complex64     # [S, D/2]
        return freqs_cis


torch.manual_seed(0)

EMBED_DIM = 256
GRID_SIZE = (24, 32)  # (H, W)
CROPS = ((0.0, 0.0), (1.5, 2.25))

cos_pt, sin_pt = get_2d_rotary_pos_embed_ref(
    EMBED_DIM,
    CROPS,
    GRID_SIZE,
    use_real=True,
    device="cuda",
)

cos, sin = ops.get_2d_rotary_pos_embed()(
    embed_dim=EMBED_DIM,
    crops_coords=CROPS,
    grid_size=GRID_SIZE,
    use_real=True,
    dtype="float32",
)

cos._attrs["name"] = "cos"
sin._attrs["name"] = "sin"
cos._attrs["is_output"] = True
sin._attrs["is_output"] = True

module = compile_model([cos, sin], detect_target(), "./tmp", "get_2d_rotary_pos_embed")

outs = module.run_with_tensors(
    {},
    {"cos": torch.empty_like(cos_pt), "sin": torch.empty_like(sin_pt)},
)

torch.testing.assert_close(outs["cos"], cos_pt, rtol=1e-5, atol=1e-5)
torch.testing.assert_close(outs["sin"], sin_pt, rtol=1e-5, atol=1e-5)


mean, _ = benchmark_module(module, count=100)

pt_mean = benchmark_torch_function(
    100,
    get_2d_rotary_pos_embed_ref,
    EMBED_DIM,
    CROPS,
    GRID_SIZE,
    use_real=True,
    device="cuda",
)
print("DinoML mean:", mean, "PT mean:", pt_mean, "speedup:", pt_mean / mean)
