import math

from honey.compiler import compile_model
from honey.frontend import IntVar, Tensor
from honey.testing import detect_target

from honey.builder.config import load_config, mark_output

batch_size = 1, 1
resolution = 512, 1024
height, width = resolution, resolution

hf_hub = "stabilityai/stable-cascade-prior"
model_name = "stable-cascade-prior"

config, honey_cls, pt_cls = load_config(hf_hub, subfolder="prior")

honey_module = honey_cls(**config)
honey_module.name_parameter_tensor()

resolution_multiple = 42.67

batch = IntVar([batch_size[0], batch_size[1]])

sample = Tensor(
    [
        batch,
        IntVar(
            [
                math.ceil(height[0] / resolution_multiple),
                math.ceil(height[1] / resolution_multiple),
            ]
        ),
        IntVar(
            [
                math.ceil(width[0] / resolution_multiple),
                math.ceil(width[1] / resolution_multiple),
            ]
        ),
        config["in_channels"],
    ],
    name="sample",
    is_input=True,
)
clip_text_pooled = Tensor(
    [batch, 1, config["clip_text_pooled_in_channels"]],
    name="clip_text_pooled",
    is_input=True,
)
clip_text = Tensor(
    [batch, IntVar([1, 77 * 16]), config["clip_text_in_channels"]],
    name="clip_text",
    is_input=True,
)
clip_img = Tensor(
    [batch, 1, config["clip_image_in_channels"]],
    name="clip_img",
    is_input=True,
)
timestep_ratio = Tensor([batch], name="timestep_ratio", is_input=True)

Y = honey_module.forward(
    sample=sample,
    timestep_ratio=timestep_ratio,
    clip_text_pooled=clip_text_pooled,
    clip_text=clip_text,
    clip_img=clip_img,
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
