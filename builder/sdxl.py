from typing import Optional

from dinoml.compiler import compile_model
from dinoml.frontend import IntVar, Tensor
from dinoml.testing import detect_target

from dinoml.builder.config import load_config, mark_output

batch_size = 1, 1
resolution = 512, 1024
height, width = resolution, resolution

hf_hub = "stabilityai/stable-diffusion-xl-base-1.0"
model_name = "stable-diffusion-xl-base-1.0"

config, dinoml_cls, pt_cls = load_config(hf_hub, subfolder="unet")

dinoml_module = dinoml_cls(**config)
dinoml_module.name_parameter_tensor()

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
# not in config 🙄
add_time_ids = Tensor(
    [IntVar([batch_size[0], batch_size[1]]), 6], name="add_time_ids", is_input=True
)
add_text_embeds = Tensor(
    [
        IntVar([batch_size[0], batch_size[1]]),
        config["projection_class_embeddings_input_dim"]
        - (6 * config["addition_time_embed_dim"]),
    ],
    name="add_text_embeds",
    is_input=True,
)
added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

Y = dinoml_module.forward(
    sample=sample,
    encoder_hidden_states=encoder_hidden_states,
    timestep=timestep,
    added_cond_kwargs=added_cond_kwargs,
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
