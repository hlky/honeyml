from typing import cast, Literal, Tuple, Union

import torch

from honey.compiler import compile_model
from honey.frontend import IntImm, IntVar, Tensor, nn
from honey.testing import detect_target
from honey.testing.benchmark_honey import benchmark_module
from honey.utils.build_utils import get_device_name, get_sm
from honey.utils.torch_utils import torch_dtype_to_string

from honey.builder.config import load_config, mark_output

from honey.mapping.autoencoder_kl import map_autoencoder_kl

def build(
    batch_size: Union[int, Tuple[int, int]],
    resolution: Union[int, Tuple[int, int]],
    hf_hub: str,
    label: str,
    model_type: Literal["decode", "encode"],
    dtype: Union[str, torch.dtype],
    device: Union[str, torch.device],
    model_name: str = "autoencoder_kl.{model_type}.{label}.{resolution}.{device_name}.sm{sm}",
    vae_scale_factor: int = 8,
    benchmark_after_compile: bool = True,
    **kwargs,
):
    device_name = get_device_name()
    sm = get_sm()
    if isinstance(batch_size, tuple):
        batch_size = IntVar(batch_size)
    elif isinstance(batch_size, int):
        batch_size = IntImm(batch_size)
    else:
        raise ValueError("`batch_size` expected `int` or `Tuple[int, int].")
    if isinstance(resolution, tuple):
        min_res, max_res = resolution
        height = IntVar([min_res // vae_scale_factor, max_res // vae_scale_factor])
        width = IntVar([min_res // vae_scale_factor, max_res // vae_scale_factor])
        resolution_label = max_res
    elif isinstance(resolution, int):
        height = IntImm(resolution)
        width = IntImm(resolution)
        resolution_label = resolution
    else:
        raise ValueError("`resolution` expected `int` or `Tuple[int, int].")
    model_name = model_name.format(
        model_type=model_type,
        label=label,
        resolution=resolution_label,
        device_name=device_name,
        sm=sm,
    )
    if isinstance(dtype, torch.dtype):
        honey_dtype = torch_dtype_to_string(dtype)
    else:
        honey_dtype = dtype
    config, honey_cls, pt_cls = load_config(hf_hub, **kwargs)
    honey_module = cast(nn.Module, honey_cls(**config, dtype=honey_dtype))
    honey_module.name_parameter_tensor()
    z = Tensor(
        [
            batch_size,
            height,
            width,
            config["latent_channels"],
        ],
        name="z",
        is_input=True,
        dtype=honey_dtype,
    )
    Y = honey_module._decode(z=z).sample
    Y = mark_output(Y, "Y")
    pt_module = pt_cls.from_pretrained(hf_hub, **kwargs)
    if model_type == "decode":
        skip_keys = ("quant.", "encoder.",)
    elif model_type == "encode":
        skip_keys = ("post_quant.", "decoder.",)
    constants = map_autoencoder_kl(
        pt_module=pt_module,
        dtype=dtype,
        device=device,
        skip_keys=skip_keys,
    )
    target = detect_target()
    module = compile_model(
        Y,
        target,
        "./tmp",
        model_name,
        constants=constants,
        dll_name=f"{model_name}.so",
    )
    if benchmark_after_compile:
        benchmark_module(module=module, count=50, repeat=3)
    return module
