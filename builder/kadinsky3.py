from dinoml.compiler import compile_model
from dinoml.frontend import IntVar, Tensor
from dinoml.testing import detect_target

from dinoml.builder.config import load_config, mark_output

batch_size = 1, 1
resolution = 512, 1024
height, width = resolution, resolution

hf_hub = "kandinsky-community/kandinsky-3"
model_name = "kandinsky-3"

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
    [1, IntVar([1, 128]), config["cross_attention_dim"]],
    name="encoder_hidden_states",
    is_input=True,
)
timestep = Tensor(
    [IntVar([batch_size[0], batch_size[1]])], name="timestep", is_input=True
)

Y = dinoml_module.forward(
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
