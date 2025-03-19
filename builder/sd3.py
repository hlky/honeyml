from honey.compiler import compile_model
from honey.frontend import IntVar, Tensor
from honey.testing import detect_target
from honey.utils.import_path import import_parent

from config import load_config, mark_output


import_parent(filepath=__file__, level=1)

import modeling

batch_size = 1, 1
resolution = 512, 1024
height, width = resolution, resolution

# TODO: update load_config for gated models
# hf_hub = "stabilityai/stable-diffusion-3-medium"
model_name = "stable-diffusion-3"
honey = modeling.transformers.SD3Transformer2DModel
# config, honey, pt = load_config(hf_hub, subfolder="unet")

honey_module = honey()
honey_module.name_parameter_tensor()

output_name = "Y"

batch = IntVar([batch_size[0], batch_size[1]])
# Workaround for name deduplication
# Model uses arange with value derived from intvar
# Codegen uses the symbolic value e.g. `48 - Y_dim_1/4`
height = IntVar([height[0] // 8, height[1] // 8], f"{output_name}_dim_1")
width = IntVar([width[0] // 8, width[1] // 8], f"{output_name}_dim_2")
hidden_states = Tensor(
    [
        batch,
        height,
        width,
        16,
    ],
    name="hidden_states",
    is_input=True,
)
encoder_hidden_states = Tensor(
    [
        batch,
        IntVar([1, 77 * 16]),
        4096,
    ],  # allow more tokens
    name="encoder_hidden_states",
    is_input=True,
)
pooled_projections = Tensor(
    [batch, 2048],
    name="pooled_projections",
    is_input=True,
)
timestep = Tensor([batch], name="timestep", is_input=True)


Y = honey_module.forward(
    hidden_states=hidden_states,
    encoder_hidden_states=encoder_hidden_states,
    timestep=timestep,
    pooled_projections=pooled_projections,
).sample
Y = mark_output(Y, output_name)

target = detect_target()
compile_model(
    Y,
    target,
    "./tmp",
    model_name,
    constants=None,
)
