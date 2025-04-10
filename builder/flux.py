from honey.compiler import compile_model
from honey.frontend import IntVar, Tensor
from honey.testing import detect_target
from honey.utils.import_path import import_parent

from honey.builder.config import load_config, mark_output

batch_size = 1
resolution = 512, 1024
height, width = resolution, resolution

model_name = "flux"
config, honey_cls, pt_cls = load_config("black-forest-labs/FLUX.1-dev")

honey_module = honey_cls(**config)
honey_module.name_parameter_tensor()

output_name = "Y"

vae_scale_factor = 16
seq_len = 512

batch = 1
height = IntVar(
    [height[0] // vae_scale_factor, height[1] // vae_scale_factor], "height"
)
width = IntVar([width[0] // vae_scale_factor, width[1] // vae_scale_factor], "width")
hidden_states = Tensor(
    [
        batch,
        height,
        width,
        config["in_channels"],
    ],
    name="hidden_states",
    is_input=True,
)
encoder_hidden_states = Tensor(
    [
        batch,
        seq_len,
        config["joint_attention_dim"],
    ],  # allow more tokens
    name="encoder_hidden_states",
    is_input=True,
)
pooled_projections = Tensor(
    [batch, config["pooled_projection_dim"]],
    name="pooled_projections",
    is_input=True,
)
timestep = Tensor([batch], name="timestep", is_input=True)
txt_ids = Tensor([batch, seq_len, 3], name="txt_ids", is_input=True)
img_ids = Tensor(
    [
        batch,
        height,
        width,
        3,
    ],
    name="img_ids",
    is_input=True,
)
guidance = (
    Tensor([batch], name="guidance", is_input=True)
    if config["guidance_embeds"]
    else None
)


Y = honey_module.forward(
    hidden_states=hidden_states,
    encoder_hidden_states=encoder_hidden_states,
    timestep=timestep,
    pooled_projections=pooled_projections,
    img_ids=img_ids,
    txt_ids=txt_ids,
    guidance=guidance,
).sample
Y = mark_output(Y, output_name)

target = detect_target()
compile_model(
    Y,
    target,
    "./tmp",
    model_name,
    constants=None,
    do_constant_folding=False,
)
