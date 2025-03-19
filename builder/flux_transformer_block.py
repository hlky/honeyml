import os
import shutil
import subprocess

import safetensors
import torch

from honey.compiler import compile_model
from honey.frontend import IntVar, Tensor
from honey.testing import benchmark_honey, detect_target
from honey.utils.import_path import import_parent
from honey.utils.misc import is_windows

from config import load_config, mark_output

import_parent(filepath=__file__, level=1)

import modeling

module_extension = ".dll" if is_windows() else ".so"
module_out_path = "H:/honey_modules/flux-dev"
weights_path = "G:/flux-dev/transformer"

constants_path = "./tmp/constants/FluxTransformerBlock"
os.makedirs(constants_path, exist_ok=True)

config, _, pt = load_config(config_file="builder/flux_dev_config.json")

vae_scale_factor = 16
seq_len = 512

weights_1 = safetensors.safe_open(
    f"{weights_path}/diffusion_pytorch_model-00001-of-00003.safetensors", "pt"
)
weights_2 = safetensors.safe_open(
    f"{weights_path}/diffusion_pytorch_model-00002-of-00003.safetensors", "pt"
)
weights_3 = safetensors.safe_open(
    f"{weights_path}/diffusion_pytorch_model-00003-of-00003.safetensors", "pt"
)


def compile(
    constants, model_name, do_compile=False, do_build=False, do_benchmark=False
):
    resolution = 768, 1024
    height, width = resolution, resolution

    honey = modeling.transformers.transformer_flux.FluxTransformerBlock
    honey_module = honey(
        dim=config["num_attention_heads"] * config["attention_head_dim"],
        num_attention_heads=config["num_attention_heads"],
        attention_head_dim=config["attention_head_dim"],
        dtype="float8_e5m2",
    )
    honey_module.name_parameter_tensor()

    batch = 1
    height = IntVar(
        [height[0] // vae_scale_factor, height[1] // vae_scale_factor], "height"
    )
    width = IntVar(
        [width[0] // vae_scale_factor, width[1] // vae_scale_factor], "width"
    )
    h_w = height * width
    hidden_states = Tensor(
        [
            batch,
            h_w,
            config["num_attention_heads"] * config["attention_head_dim"],
        ],
        name="hidden_states",
        is_input=True,
    )
    encoder_hidden_states = Tensor(
        [
            batch,
            seq_len,
            config["num_attention_heads"] * config["attention_head_dim"],
        ],
        name="encoder_hidden_states",
        is_input=True,
    )
    temb = Tensor(
        [batch, config["num_attention_heads"] * config["attention_head_dim"]],
        name="temb",
        is_input=True,
    )
    image_rotary_emb = Tensor(
        [batch, 1, h_w + seq_len, 64, 2, 2],  # check 64
        name="image_rotary_emb",
        is_input=True,
        dtype="float32",
    )

    encoder_hidden_states_out, hidden_states_out = honey_module.forward(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        temb=temb,
        image_rotary_emb=image_rotary_emb,
    )
    encoder_hidden_states_out = mark_output(
        encoder_hidden_states_out, "encoder_hidden_states_out"
    )
    hidden_states_out = mark_output(hidden_states_out, "hidden_states_out")
    target = detect_target()
    module = compile_model(
        [encoder_hidden_states_out, hidden_states_out],
        target,
        "./tmp",
        model_name,
        dll_name=model_name + module_extension,
        constants=constants,
        do_constant_folding=False,
        do_compile=do_compile,
        do_build=do_build,
    )
    if do_benchmark:
        benchmark_honey.benchmark_module(module)


for block_idx in range(0, config["num_layers"]):
    prefix = "transformer_blocks"
    constants = {}
    block_prefix = f"{prefix}.{block_idx}."
    model_name = f"FluxTransformerBlock.{block_idx}"
    if block_idx != 0 and os.path.exists(f"{constants_path}/{model_name + '.bin'}"):
        continue
    for weights in [weights_1, weights_2, weights_3]:
        for key in weights.keys():
            if not key.startswith(block_prefix):
                continue
            skip = [
                ".add_v_proj.",
                ".add_k_proj.",
                ".add_q_proj.",
            ]
            constants[key.replace(block_prefix, "").replace(".", "_")] = (
                weights.get_tensor(key).to(torch.float16)
                if any([skip_key in key for skip_key in skip])
                else weights.get_tensor(key).to(torch.float8_e5m2)
            )
    compile(constants, model_name, do_compile=block_idx == 0)
    shutil.move(
        f"./tmp/{model_name}/constants.bin",
        f"{constants_path}/{model_name + '.bin'}",
    )
    if block_idx == 0:
        shutil.move(f"./tmp/{model_name}", "./tmp/FluxTransformerBlock")
    else:
        shutil.rmtree(f"./tmp/{model_name}")

for block_idx in range(0, config["num_layers"]):
    model_name = f"FluxTransformerBlock.{block_idx}"
    shutil.copy(
        f"{constants_path}/{model_name + '.bin'}",
        f"./tmp/FluxTransformerBlock/constants.bin",
    )
    if block_idx == 0:
        cmake = 'cmake -B "tmp/FluxTransformerBlock/build" -S "tmp/FluxTransformerBlock" -G "Visual Studio 17 2022" -A x64'
        process = subprocess.Popen(cmake)
        process.wait()
    command = 'msbuild "tmp/FluxTransformerBlock/build/FluxTransformerBlock.0.sln" -m /property:Configuration=Release'
    process = subprocess.Popen(command)
    process.wait()
    shutil.move(
        f"./tmp/FluxTransformerBlock/build/Release/model.dll",
        f"{module_out_path}/{model_name + module_extension}",
    )
