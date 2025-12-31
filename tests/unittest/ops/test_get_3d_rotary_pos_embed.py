from typing import Optional, Tuple, Union
import torch

from dinoml.compiler import compile_model, ops
from dinoml.testing import detect_target
from dinoml.testing.benchmark_dinoml import benchmark_module
from dinoml.testing.benchmark_pt import benchmark_torch_function


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
    is_npu = freqs.device.type == "npu"
    if is_npu:
        freqs = freqs.float()
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


def get_3d_rotary_pos_embed_ref(
    embed_dim,
    crops_coords,
    grid_size,
    temporal_size,
    theta: int = 10000,
    use_real: bool = True,
    grid_type: str = "linspace",
    max_size: Optional[Tuple[int, int]] = None,
    device: Optional[torch.device] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    RoPE for video tokens with 3D structure.

    Args:
    embed_dim: (`int`):
        The embedding dimension size, corresponding to hidden_size_head.
    crops_coords (`Tuple[int]`):
        The top-left and bottom-right coordinates of the crop.
    grid_size (`Tuple[int]`):
        The grid size of the spatial positional embedding (height, width).
    temporal_size (`int`):
        The size of the temporal dimension.
    theta (`float`):
        Scaling factor for frequency computation.
    grid_type (`str`):
        Whether to use "linspace" or "slice" to compute grids.

    Returns:
        `torch.Tensor`: positional embedding with shape `(temporal_size * grid_size[0] * grid_size[1], embed_dim/2)`.
    """
    if use_real is not True:
        raise ValueError(
            " `use_real = False` is not currently supported for get_3d_rotary_pos_embed"
        )

    if grid_type == "linspace":
        start, stop = crops_coords
        grid_size_h, grid_size_w = grid_size
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
        grid_t = torch.arange(temporal_size, device=device, dtype=torch.float32)
        grid_t = torch.linspace(
            0,
            temporal_size * (temporal_size - 1) / temporal_size,
            temporal_size,
            device=device,
            dtype=torch.float32,
        )
    elif grid_type == "slice":
        max_h, max_w = max_size
        grid_size_h, grid_size_w = grid_size
        grid_h = torch.arange(max_h, device=device, dtype=torch.float32)
        grid_w = torch.arange(max_w, device=device, dtype=torch.float32)
        grid_t = torch.arange(temporal_size, device=device, dtype=torch.float32)
    else:
        raise ValueError("Invalid value passed for `grid_type`.")

    # Compute dimensions for each axis
    dim_t = embed_dim // 4
    dim_h = embed_dim // 8 * 3
    dim_w = embed_dim // 8 * 3

    # Temporal frequencies
    freqs_t = get_1d_rotary_pos_embed(dim_t, grid_t, theta=theta, use_real=True)
    # Spatial frequencies for height and width
    freqs_h = get_1d_rotary_pos_embed(dim_h, grid_h, theta=theta, use_real=True)
    freqs_w = get_1d_rotary_pos_embed(dim_w, grid_w, theta=theta, use_real=True)

    # BroadCast and concatenate temporal and spaial frequencie (height and width) into a 3d tensor
    def combine_time_height_width(freqs_t, freqs_h, freqs_w):
        freqs_t = freqs_t[:, None, None, :].expand(
            -1, grid_size_h, grid_size_w, -1
        )  # temporal_size, grid_size_h, grid_size_w, dim_t
        freqs_h = freqs_h[None, :, None, :].expand(
            temporal_size, -1, grid_size_w, -1
        )  # temporal_size, grid_size_h, grid_size_2, dim_h
        freqs_w = freqs_w[None, None, :, :].expand(
            temporal_size, grid_size_h, -1, -1
        )  # temporal_size, grid_size_h, grid_size_2, dim_w

        freqs = torch.cat(
            [freqs_t, freqs_h, freqs_w], dim=-1
        )  # temporal_size, grid_size_h, grid_size_w, (dim_t + dim_h + dim_w)
        freqs = freqs.view(
            temporal_size * grid_size_h * grid_size_w, -1
        )  # (temporal_size * grid_size_h * grid_size_w), (dim_t + dim_h + dim_w)
        return freqs

    t_cos, t_sin = freqs_t  # both t_cos and t_sin has shape: temporal_size, dim_t
    h_cos, h_sin = freqs_h  # both h_cos and h_sin has shape: grid_size_h, dim_h
    w_cos, w_sin = freqs_w  # both w_cos and w_sin has shape: grid_size_w, dim_w

    if grid_type == "slice":
        t_cos, t_sin = t_cos[:temporal_size], t_sin[:temporal_size]
        h_cos, h_sin = h_cos[:grid_size_h], h_sin[:grid_size_h]
        w_cos, w_sin = w_cos[:grid_size_w], w_sin[:grid_size_w]

    cos = combine_time_height_width(t_cos, h_cos, w_cos)
    sin = combine_time_height_width(t_sin, h_sin, w_sin)
    return cos, sin


torch.manual_seed(0)

# Use an embed_dim divisible by 16 (required so axis dims are even)
EMBED_DIM = 128
GRID_SIZE = (12, 10)  # (H, W)
TEMPORAL = 6
THETA = 10000

# -------- test linspace --------
CROPS = ((0.0, 0.0), (1.25, 2.0))

cos_pt, sin_pt = get_3d_rotary_pos_embed_ref(
    EMBED_DIM,
    CROPS,
    GRID_SIZE,
    TEMPORAL,
    theta=THETA,
    use_real=True,
    grid_type="linspace",
    max_size=None,
    device="cuda",
)

cos, sin = ops.get_3d_rotary_pos_embed()(
    embed_dim=EMBED_DIM,
    crops_coords=CROPS,
    grid_size=GRID_SIZE,
    temporal_size=TEMPORAL,
    theta=THETA,
    use_real=True,
    grid_type="linspace",
    max_size=None,
    device=None,
    dtype="float32",
)

cos._attrs["name"] = "cos"
sin._attrs["name"] = "sin"
cos._attrs["is_output"] = True
sin._attrs["is_output"] = True

module = compile_model(
    [cos, sin], detect_target(), "./tmp", "get_3d_rotary_pos_embed_linspace"
)

outs = module.run_with_tensors(
    {},
    {"cos": torch.empty_like(cos_pt), "sin": torch.empty_like(sin_pt)},
)
torch.testing.assert_close(outs["cos"], cos_pt, rtol=1e-5, atol=1e-5)
torch.testing.assert_close(outs["sin"], sin_pt, rtol=1e-5, atol=1e-5)


mean, _ = benchmark_module(module, count=100)

pt_mean = benchmark_torch_function(
    100,
    get_3d_rotary_pos_embed_ref,
    EMBED_DIM,
    CROPS,
    GRID_SIZE,
    TEMPORAL,
    theta=THETA,
    use_real=True,
    grid_type="linspace",
    max_size=None,
    device="cuda",
)
print("DinoML mean:", mean, "PT mean:", pt_mean, "speedup:", pt_mean / mean)


# -------- test slice --------
MAX_SIZE = (24, 24)

cos_pt2, sin_pt2 = get_3d_rotary_pos_embed_ref(
    EMBED_DIM,
    CROPS,  # crops unused in slice path; keep API the same
    GRID_SIZE,
    TEMPORAL,
    theta=THETA,
    use_real=True,
    grid_type="slice",
    max_size=MAX_SIZE,
    device="cuda",
)

cos2, sin2 = ops.get_3d_rotary_pos_embed()(
    embed_dim=EMBED_DIM,
    crops_coords=CROPS,
    grid_size=GRID_SIZE,
    temporal_size=TEMPORAL,
    theta=THETA,
    use_real=True,
    grid_type="slice",
    max_size=MAX_SIZE,
    dtype="float32",
)

cos2._attrs["name"] = "cos"
sin2._attrs["name"] = "sin"
cos2._attrs["is_output"] = True
sin2._attrs["is_output"] = True

module2 = compile_model(
    [cos2, sin2], detect_target(), "./tmp", "get_3d_rotary_pos_embed_slice"
)

outs2 = module2.run_with_tensors(
    {},
    {"cos": torch.empty_like(cos_pt2), "sin": torch.empty_like(sin_pt2)},
)
torch.testing.assert_close(outs2["cos"], cos_pt2, rtol=1e-5, atol=1e-5)
torch.testing.assert_close(outs2["sin"], sin_pt2, rtol=1e-5, atol=1e-5)

mean, _ = benchmark_module(module, count=100)

pt_mean = benchmark_torch_function(
    100,
    get_3d_rotary_pos_embed_ref,
    EMBED_DIM,
    CROPS,  # crops unused in slice path; keep API the same
    GRID_SIZE,
    TEMPORAL,
    theta=THETA,
    use_real=True,
    grid_type="slice",
    max_size=MAX_SIZE,
    device="cuda",
)
print("DinoML mean:", mean, "PT mean:", pt_mean, "speedup:", pt_mean / mean)
