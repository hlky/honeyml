from typing import Union
import torch

from dinoml.compiler import compile_model, ops
from dinoml.frontend import Tensor
from dinoml.testing import detect_target
from dinoml.testing.benchmark_dinoml import benchmark_module
from dinoml.testing.benchmark_pt import benchmark_torch_function

TOLERANCE = 3e-5


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
        return freqs_cis.real.float(), freqs_cis.imag.float()


def run_case(dim, pos_kind: str, use_real: bool, repeat_interleave_real: bool):
    theta = 10000.0
    linear_factor = 1.25
    ntk_factor = 1.1

    if pos_kind == "int":
        pos = 257
        ref0, ref1 = get_1d_rotary_pos_embed(
            dim,
            pos,
            theta=theta,
            use_real=use_real,
            linear_factor=linear_factor,
            ntk_factor=ntk_factor,
            repeat_interleave_real=repeat_interleave_real,
            device="cuda",
        )
        out0, out1 = ops.get_1d_rotary_pos_embed()(
            dim=dim,
            pos=pos,
            theta=theta,
            use_real=use_real,
            linear_factor=linear_factor,
            ntk_factor=ntk_factor,
            repeat_interleave_real=repeat_interleave_real,
            dtype="float32",
        )
        out0._attrs["name"] = "out0"
        out1._attrs["name"] = "out1"
        out0._attrs["is_output"] = True
        out1._attrs["is_output"] = True

        mod = compile_model(
            [out0, out1],
            detect_target(),
            "./tmp",
            f"get_1d_rope_int_{use_real}_{repeat_interleave_real}",
        )
        outs = mod.run_with_tensors(
            {},
            {
                "out0": torch.empty_like(ref0).contiguous(),
                "out1": torch.empty_like(ref1).contiguous(),
            },
        )

        torch.testing.assert_close(outs["out0"], ref0, rtol=TOLERANCE, atol=TOLERANCE)
        torch.testing.assert_close(outs["out1"], ref1, rtol=TOLERANCE, atol=TOLERANCE)

        mean, _ = benchmark_module(mod, count=100)
        pt_mean = benchmark_torch_function(
            100,
            get_1d_rotary_pos_embed,
            dim,
            pos,
            theta=theta,
            use_real=use_real,
            linear_factor=linear_factor,
            ntk_factor=ntk_factor,
            repeat_interleave_real=repeat_interleave_real,
            device="cuda",
        )
        print("DinoML mean:", mean, "PT mean:", pt_mean, "speedup:", pt_mean / mean)

    elif pos_kind == "tensor":
        S = 193
        pos_pt = torch.linspace(0.0, 127.5, steps=S, device="cuda", dtype=torch.float32)
        ref0, ref1 = get_1d_rotary_pos_embed(
            dim,
            pos_pt,
            theta=theta,
            use_real=use_real,
            linear_factor=linear_factor,
            ntk_factor=ntk_factor,
            repeat_interleave_real=repeat_interleave_real,
        )

        pos = Tensor([S], name="pos", is_input=True, dtype="float32")
        out0, out1 = ops.get_1d_rotary_pos_embed()(
            dim=dim,
            pos=pos,
            theta=theta,
            use_real=use_real,
            linear_factor=linear_factor,
            ntk_factor=ntk_factor,
            repeat_interleave_real=repeat_interleave_real,
            dtype="float32",
        )
        out0._attrs["name"] = "out0"
        out1._attrs["name"] = "out1"
        out0._attrs["is_output"] = True
        out1._attrs["is_output"] = True

        mod = compile_model(
            [out0, out1],
            detect_target(),
            "./tmp",
            f"get_1d_rope_tensor_{use_real}_{repeat_interleave_real}",
        )
        outs = mod.run_with_tensors(
            {"pos": pos_pt.contiguous()},
            {
                "out0": torch.empty_like(ref0).contiguous(),
                "out1": torch.empty_like(ref1).contiguous(),
            },
        )

        torch.testing.assert_close(outs["out0"], ref0, rtol=TOLERANCE, atol=TOLERANCE)
        torch.testing.assert_close(outs["out1"], ref1, rtol=TOLERANCE, atol=TOLERANCE)

        mean, _ = benchmark_module(mod, count=100)
        pt_mean = benchmark_torch_function(
            100,
            get_1d_rotary_pos_embed,
            dim,
            pos_pt,
            theta=theta,
            use_real=use_real,
            linear_factor=linear_factor,
            ntk_factor=ntk_factor,
            repeat_interleave_real=repeat_interleave_real,
        )
        print("DinoML mean:", mean, "PT mean:", pt_mean, "speedup:", pt_mean / mean)

    else:
        raise RuntimeError(pos_kind)


torch.manual_seed(0)

# dim must be even; use a few different sizes
for dim in [64, 128, 256]:
    # use_real=False (repeat_interleave_real irrelevant)
    run_case(dim, "int", use_real=False, repeat_interleave_real=True)
    run_case(dim, "tensor", use_real=False, repeat_interleave_real=True)

    # use_real=True, repeat_interleave_real=True
    run_case(dim, "int", use_real=True, repeat_interleave_real=True)
    run_case(dim, "tensor", use_real=True, repeat_interleave_real=True)

    # use_real=True, repeat_interleave_real=False
    run_case(dim, "int", use_real=True, repeat_interleave_real=False)
    run_case(dim, "tensor", use_real=True, repeat_interleave_real=False)
