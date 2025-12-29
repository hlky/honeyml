import torch
import torch.nn
import math


from dinoml.compiler import compile_model, ops
from dinoml.frontend import IntImm, IntVar, Tensor, nn
from dinoml.testing import detect_target
from dinoml.testing.benchmark_dinoml import benchmark_module
from dinoml.testing.benchmark_pt import benchmark_torch_function


def gelu_new(input: torch.Tensor) -> torch.Tensor:
    return (
        0.5
        * input
        * (
            1.0
            + torch.tanh(
                math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))
            )
        )
    )


torch.manual_seed(0)

x_pt = torch.randn([1, 512, 4096]).to("cuda", torch.bfloat16)
y_pt = gelu_new(x_pt.clone())

x = Tensor([1, 512, 4096], name="x", is_input=True, dtype="bfloat16")
y = ops.gelu_new()(x)
y._attrs["name"] = "y"
y._attrs["is_output"] = True

module = compile_model(y, detect_target(), "./tmp", "gelu_new")

out = module.run_with_tensors({"x": x_pt}, {"y": torch.empty_like(y_pt)})["y"]

torch.testing.assert_close(out, y_pt, rtol=1e-5, atol=1e-5)

benchmark_module(module, count=100)

pt_mean = benchmark_torch_function(100, gelu_new, x_pt)
print(pt_mean)
