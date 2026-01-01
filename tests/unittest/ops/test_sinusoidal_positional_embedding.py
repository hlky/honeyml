import math
import torch
from dinoml.compiler import compile_model, ops
from dinoml.frontend import Tensor
from dinoml.testing import detect_target
from dinoml.testing.benchmark_dinoml import benchmark_module
from dinoml.testing.benchmark_pt import benchmark_torch_function


def sinusoidal_positional_embedding(x, max_seq_length: int = 32):
    B, L, embed_dim = x.shape
    position = torch.arange(max_seq_length, device=x.device, dtype=x.dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2, device=x.device, dtype=x.dtype)
        * (-math.log(10000.0) / embed_dim)
    )
    pe = torch.zeros(1, max_seq_length, D, device=x.device, dtype=x.dtype)
    pe[:, :, 0::2] = torch.sin(position * div_term)
    pe[:, :, 1::2] = torch.cos(position * div_term)
    return x + pe[:, :L]


for dtype in ["float32"]:
    for dim in [320, 640, 1280]:
        for seq_len in [16, 32]:
            if dtype == "float32":
                torch_dtype = torch.float32
            else:
                torch_dtype = torch.float16
            B, L, D = 2, seq_len, dim
            x = torch.randn(B, L, D, device="cuda", dtype=torch_dtype)

            y_ref = sinusoidal_positional_embedding(x)

            x_t = Tensor([B, L, D], dtype=dtype, name="x", is_input=True)
            y = ops.sinusoidal_positional_embedding()(x_t, D, 32)
            y._attrs["name"] = "y"
            y._attrs["is_output"] = True

            module = compile_model(
                y,
                detect_target(),
                "./tmp",
                f"test_sinusoidal_positional_embedding_{dtype}_{dim}_{seq_len}",
            )
            out = module.run_with_tensors(
                {"x": x.contiguous()}, {"y": torch.empty_like(y_ref).contiguous()}
            )["y"]

            torch.testing.assert_close(out, y_ref, rtol=1e-5, atol=1e-5)

            mean, _ = benchmark_module(module, count=100)
            pt_mean = benchmark_torch_function(100, sinusoidal_positional_embedding, x)
            print("DinoML mean:", mean, "PT mean:", pt_mean, "speedup:", pt_mean / mean)
