from typing import Union
import torch

from dinoml.compiler import compile_model, ops
from dinoml.testing import detect_target
from dinoml.testing.benchmark_dinoml import benchmark_module
from dinoml.testing.benchmark_pt import benchmark_torch_function


def get_2d_rotary_pos_embed_lumina(
    embed_dim, len_h, len_w, linear_factor=1.0, ntk_factor=1.0, device=None
):
    assert embed_dim % 4 == 0

    emb_h = get_1d_rotary_pos_embed(
        embed_dim // 2,
        len_h,
        linear_factor=linear_factor,
        ntk_factor=ntk_factor,
        device=device,
    )  # (H, D/4)
    emb_w = get_1d_rotary_pos_embed(
        embed_dim // 2,
        len_w,
        linear_factor=linear_factor,
        ntk_factor=ntk_factor,
        device=device,
    )  # (W, D/4)
    emb_h = emb_h.view(len_h, 1, embed_dim // 4, 1).repeat(
        1, len_w, 1, 1
    )  # (H, W, D/4, 1)
    emb_w = emb_w.view(1, len_w, embed_dim // 4, 1).repeat(
        len_h, 1, 1, 1
    )  # (H, W, D/4, 1)

    emb = torch.cat([emb_h, emb_w], dim=-1).flatten(2)  # (H, W, D/2)
    return emb


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[torch.Tensor, int],
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype=torch.float32,  #  torch.float32, torch.float64 (flux)
    device=None,
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

    if isinstance(pos, int):
        pos = torch.arange(pos, device=device)

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
LEN_H = 24
LEN_W = 32
LINEAR = 1
NTK = 1

ref = get_2d_rotary_pos_embed_lumina(
    EMBED_DIM, LEN_H, LEN_W, linear_factor=LINEAR, ntk_factor=NTK, device="cuda"
)
ref_real = ref.real.float()
ref_imag = ref.imag.float()

real, imag = ops.get_2d_rotary_pos_embed_lumina()(
    embed_dim=EMBED_DIM,
    len_h=LEN_H,
    len_w=LEN_W,
    linear_factor=LINEAR,
    ntk_factor=NTK,
    dtype="float32",
)
real._attrs["name"] = "real"
imag._attrs["name"] = "imag"
real._attrs["is_output"] = True
imag._attrs["is_output"] = True

module = compile_model(
    [real, imag], detect_target(), "./tmp", "get_2d_rotary_pos_embed_lumina"
)

outs = module.run_with_tensors(
    {},
    {
        "real": torch.empty_like(ref_real).contiguous(),
        "imag": torch.empty_like(ref_imag).contiguous(),
    },
)

torch.testing.assert_close(outs["real"], ref_real, rtol=1e-5, atol=1e-5)
torch.testing.assert_close(outs["imag"], ref_imag, rtol=1e-5, atol=1e-5)

mean, _ = benchmark_module(module, count=100)
pt_mean = benchmark_torch_function(
    100,
    get_2d_rotary_pos_embed_lumina,
    EMBED_DIM,
    LEN_H,
    LEN_W,
    linear_factor=LINEAR,
    ntk_factor=NTK,
    device="cuda",
)
print("DinoML mean:", mean, "PT mean:", pt_mean, "speedup:", pt_mean / mean)
