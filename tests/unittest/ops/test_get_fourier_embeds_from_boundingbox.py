import torch

from dinoml.compiler import compile_model, ops
from dinoml.frontend import Tensor
from dinoml.testing import detect_target
from dinoml.testing.benchmark_dinoml import benchmark_module
from dinoml.testing.benchmark_pt import benchmark_torch_function


def ref_get_fourier_embeds_from_boundingbox(
    embed_dim: int, box: torch.Tensor
) -> torch.Tensor:
    B, N = box.shape[:2]
    emb = 100 ** (
        torch.arange(embed_dim, device=box.device, dtype=torch.float32) / embed_dim
    )
    emb = emb[None, None, None].to(device=box.device, dtype=box.dtype)
    emb = emb * box.unsqueeze(-1)  # [B, N, 4, E]
    emb = torch.stack((emb.sin(), emb.cos()), dim=-1)  # [B, N, 4, E, 2]
    emb = emb.permute(0, 1, 3, 4, 2).reshape(
        B, N, embed_dim * 2 * 4
    )  # [B, N, E, 2, 4] -> [B,N,E*8]
    return emb


torch.manual_seed(0)

B = 2
N = 7
E = 64

box_pt = (torch.randn(B, N, 4, device="cuda", dtype=torch.float16) * 0.5).contiguous()
y_ref = ref_get_fourier_embeds_from_boundingbox(E, box_pt)

box = Tensor([B, N, 4], name="box", is_input=True, dtype="float16")
y = ops.get_fourier_embeds_from_boundingbox()(E, box)
y._attrs["name"] = "y"
y._attrs["is_output"] = True

module = compile_model(
    y, detect_target(), "./tmp", "get_fourier_embeds_from_boundingbox"
)

out = module.run_with_tensors(
    {"box": box_pt}, {"y": torch.empty_like(y_ref).contiguous()}
)["y"]

torch.testing.assert_close(out, y_ref, rtol=5e-4, atol=5e-4)

mean, _ = benchmark_module(module, count=100)
pt_mean = benchmark_torch_function(
    100, ref_get_fourier_embeds_from_boundingbox, E, box_pt
)
print("DinoML mean:", mean, "ms")
print("PyTorch mean:", pt_mean, "ms")
print(f"DinoML is {pt_mean / mean:.2f}x PyTorch")
