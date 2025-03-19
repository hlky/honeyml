from typing import Optional

from honey.compiler import compile_model
from honey.frontend import IntVar, Tensor
from honey.testing import detect_target

from config import load_config, mark_output

batch_size = 1, 1
resolution = 512, 1024
height, width = resolution, resolution

hf_hub = "runwayml/stable-diffusion-inpainting"
model_name = "stable-diffusion-inpainting"

config, honey, pt = load_config(hf_hub, subfolder="unet")

honey_module = honey(**config)
honey_module.name_parameter_tensor()

sample = Tensor(
    [
        IntVar([batch_size[0], batch_size[1]]),
        IntVar([height[0] // 8, height[1] // 8]),
        IntVar([width[0] // 8, width[1] // 8]),
        config["in_channels"],
    ],
    name="sample",
    is_input=True,
)
encoder_hidden_states = Tensor(
    [1, IntVar([1, 77 * 16]), config["cross_attention_dim"]],  # allow more tokens
    name="encoder_hidden_states",
    is_input=True,
)
timestep = Tensor(
    [IntVar([batch_size[0], batch_size[1]])], name="timestep", is_input=True
)

Y = honey_module.forward(
    sample=sample,
    encoder_hidden_states=encoder_hidden_states,
    timestep=timestep,
).sample
Y = mark_output(Y, "Y")

target = detect_target()
compile_model(
    Y,
    target,
    "./tmp",
    model_name,
    constants=None,
)
