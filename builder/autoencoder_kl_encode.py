from dinoml.compiler import compile_model
from dinoml.frontend import IntVar, Tensor
from dinoml.testing import detect_target
from dinoml.testing.benchmark_dinoml import benchmark_module
from dinoml.utils.build_utils import get_device_name, get_sm

from config import load_config, mark_output

import torch

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--max", type=int)
parser.add_argument("--min", type=int, default=8)
parser.add_argument("--min-batch", type=int, default=1)
parser.add_argument("--max-batch", type=int, default=1)
parser.add_argument("--hf_hub", type=str, default="runwayml/stable-diffusion-v1-5")
parser.add_argument("--label", type=str, default="v1")
parser.add_argument(
    "--subfolder", type=str, default=None, help="`vae` if `hf_hub` is for a pipeline."
)
parser.add_argument("--dtype", type=str, default="float16")

args = parser.parse_args()


def torch_dtype_from_str(dtype: str):
    return torch.__dict__.get(dtype, None)


def map_vae(pt_module, device="cuda", dtype="float16", encoder=True):
    if not isinstance(pt_module, dict):
        pt_params = dict(pt_module.named_parameters())
    else:
        pt_params = pt_module
    params_dinoml = {}
    quant_key = "post_quant" if encoder else "quant"
    vae_key = "decoder" if encoder else "encoder"
    for key, arr in pt_params.items():
        if key.startswith(vae_key):
            continue
        if key.startswith(quant_key):
            continue
        arr = arr.to(device, dtype=torch_dtype_from_str(dtype))
        key = key.replace(".", "_")
        if (
            "conv" in key
            and "norm" not in key
            and key.endswith("_weight")
            and len(arr.shape) == 4
        ):
            params_dinoml[key] = torch.permute(arr, [0, 2, 3, 1]).contiguous()
        else:
            params_dinoml[key] = arr
    if encoder:
        params_dinoml["encoder_conv_in_weight"] = torch.functional.F.pad(
            params_dinoml["encoder_conv_in_weight"], (0, 1, 0, 0, 0, 0, 0, 0)
        )

    return params_dinoml


device_name = get_device_name()

sm = get_sm()


batch_size = args.min_batch, args.max_batch
resolution = args.min, args.max
height, width = resolution, resolution

hf_hub = args.hf_hub
label = args.label
model_name = f"autoencoder_kl.encoder.{label}.{resolution[1]}.{device_name}.sm{sm}"

config, dinoml_cls, pt_cls = load_config(hf_hub, subfolder=args.subfolder)

dinoml_module = dinoml_cls(**config)
dinoml_module.name_parameter_tensor()

x = Tensor(
    [
        IntVar([batch_size[0], batch_size[1]]),
        IntVar([height[0], height[1]]),
        IntVar([width[0], width[1]]),
        config["out_channels"],
    ],
    name="x",
    is_input=True,
    dtype=args.dtype,
)
sample = Tensor(
    [
        IntVar([batch_size[0], batch_size[1]]),
        IntVar([height[0] // 8, height[1] // 8]),
        IntVar([width[0] // 8, width[1] // 8]),
        config["latent_channels"],
    ],
    name="sample",
    is_input=True,
    dtype=args.dtype,
)

latents = dinoml_module.encode(x=x).latent_dist.sample(sample)
latents = mark_output(latents, "latents")

pt = pt_cls.from_pretrained(hf_hub, subfolder=args.subfolder)
constants = map_vae(pt)

target = detect_target()

module = compile_model(
    latents,
    target,
    "./tmp",
    model_name,
    constants=constants,
    dll_name=f"{model_name}.so",
)

benchmark_module(module=module, count=50, repeat=3)
