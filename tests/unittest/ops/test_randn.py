import torch

from dinoml.compiler import compile_model, ops
from dinoml.frontend import IntImm, IntVar, Tensor, nn
from dinoml.testing import detect_target
from dinoml.utils.build_utils import (
    get_device_name,
    get_sm,
)

from dinoml.testing.benchmark_dinoml import benchmark_module, prepare_inputs_outputs

from dinoml.builder.config import mark_output

device_name = get_device_name()

sm = get_sm()

model_name = f"randn_test.{device_name}.sm{sm}"

dinoml_module = ops.randn([1, 4, 64, 64], dtype="float16", seed=69)
pt_output = torch.randn(
    [1, 4, 64, 64],
    generator=torch.Generator("cuda").manual_seed(69),
    dtype=torch.float16,
    device="cuda",
)

Y = dinoml_module()
Y = mark_output(Y, "Y")

target = detect_target()

module = compile_model(
    Y,
    target,
    "./tmp",
    model_name,
    dll_name=f"{model_name}.so",
)

inputs, outputs = prepare_inputs_outputs(module)
output = module.run_with_tensors(inputs, outputs)["Y"]

print(f"{output=}")

print(f"{pt_output=}")

torch.testing.assert_close(pt_output, output)

benchmark_module(module=module, count=50, repeat=3)


model_name = f"randn_test_seed_repeat.{device_name}.sm{sm}"

dinoml_module = ops.randn([1, 4, 64, 64], dtype="float16", seed=69)
pt_output = torch.randn(
    [1, 4, 64, 64],
    generator=torch.Generator("cuda").manual_seed(69),
    dtype=torch.float16,
    device="cuda",
)

Y = dinoml_module()
Y = mark_output(Y, "Y")

target = detect_target()

module = compile_model(
    Y,
    target,
    "./tmp",
    model_name,
    dll_name=f"{model_name}.so",
)

for _ in range(5):
    inputs, outputs = prepare_inputs_outputs(module)
    output = module.run_with_tensors(inputs, outputs)["Y"]

    print(f"{output=}")

    print(f"{pt_output=}")

    torch.testing.assert_close(pt_output, output)

model_name = f"randn_test_multi.{device_name}.sm{sm}"
dinoml_module = ops.randn([1, 4, 64, 64], dtype="float16")

Y = dinoml_module()
Y = mark_output(Y, "Y")

target = detect_target()

module = compile_model(
    Y,
    target,
    "./tmp",
    model_name,
    dll_name=f"{model_name}.so",
)

for _ in range(5):
    inputs, outputs = prepare_inputs_outputs(module)
    output = module.run_with_tensors(inputs, outputs)["Y"]
    print(f"{output=}")


model_name = f"randn_test_dynamic.{device_name}.sm{sm}"

height, width = IntVar([8, 64]), IntVar([8, 64])

X = Tensor([1, 4, height, width], is_input=True)

dinoml_module = ops.randn([1, 4, height, width], dtype="float16")

Y = X + dinoml_module()
Y = mark_output(Y, "Y")

target = detect_target()

module = compile_model(
    Y,
    target,
    "./tmp",
    model_name,
    dll_name=f"{model_name}.so",
)

for _ in range(5):
    inputs, outputs = prepare_inputs_outputs(module, use_zeros=True)
    output = module.run_with_tensors(inputs, outputs)["Y"]
    print(f"{output=}")
