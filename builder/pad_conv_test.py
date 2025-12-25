from honey.compiler import compile_model, ops
from honey.frontend import IntVar, Tensor, nn
from honey.testing import detect_target
from honey.testing.benchmark_honey import benchmark_module

from honey.builder.config import load_config, mark_output

import torch


def torch_dtype_from_str(dtype: str):
    return torch.__dict__.get(dtype, None)


device_name = (
    torch.cuda.get_device_name()
    .lower()
    .replace("nvidia ", "")
    .replace("geforce rtx ", "")
    .replace("geforce gtx ", "")
    .replace("geforce gt ", "")
    .replace("geforce ", "")
    .replace("tesla ", "")
    .replace("quadro ", "")
    .strip()
    .replace(" ", "_")
    .lower()
    .split(",")[0]
    .split("(")[0]
)

sm = "".join(str(i) for i in torch.cuda.get_device_capability())

model_name = f"pad_conv_test.{device_name}.sm{sm}"

x = Tensor(
    [
        IntVar([1, 1]),
        IntVar([64, 512]),
        IntVar([64, 512]),
        4,
    ],
    name="x",
    is_input=True,
)


class PadConvTestPT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            4,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def __call__(
        self,
        x: torch.Tensor,
    ):
        x = torch.nn.functional.pad(x, pad=(0, 1, 0, 1), mode="constant", value=0.0)
        x = self.conv(x)
        return x


class PadConvTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            4,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def __call__(
        self,
        x: Tensor,
    ):
        x = ops.pad(pad=(0, 1, 0, 1), mode="constant", value=0.0)(x)
        x = self.conv(x)
        return x


honey_module = PadConvTest()
pt_module = PadConvTestPT()

constants = dict(pt_module.state_dict())
constants["conv.weight"] = (
    constants["conv.weight"].permute(0, 2, 3, 1).contiguous().cuda().to(torch.float16)
)

Y = honey_module(x)
Y = mark_output(Y, "Y")

target = detect_target()

module = compile_model(
    Y,
    target,
    "./tmp",
    model_name,
    constants=constants,
    dll_name=f"{model_name}.so",
)

benchmark_module(module=module, count=50, repeat=3)
