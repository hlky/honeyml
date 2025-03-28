from typing import Optional, cast, Tuple, Union

import torch

from honey.compiler import compile_model
from honey.frontend import IntImm, IntVar, Tensor, nn
from honey.mapping.unet2d_condition import map_unet2d_condition
from honey.testing import detect_target
from honey.testing.benchmark_honey import benchmark_module
from honey.utils.build_utils import get_device_name, get_sm
from honey.utils.torch_utils import torch_dtype_to_string

from honey.builder.config import load_config, mark_output


def build(
    batch_size: Union[int, Tuple[int, int]],
    resolution: Union[int, Tuple[int, int]],
    hf_hub: str,
    label: str,
    dtype: Union[str, torch.dtype],
    device: Union[str, torch.device],
    model_name: str = "unet2d_condition.{label}.{resolution}.{device_name}.sm{sm}",
    vae_scale_factor: int = 8,
    benchmark_after_compile: bool = True,
    extend_seq_len_by_multiple: int = 16,
    store_constants_in_module: bool = True,
    num_add_time_ids: Optional[int] = None,
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
        label=label,
        resolution=resolution_label,
        device_name=device_name,
        sm=sm,
    )
    config, honey_cls, pt_cls = load_config(hf_hub, **kwargs)
    honey_module = cast(nn.Module, honey_cls(**config))
    honey_module.name_parameter_tensor()
    if isinstance(dtype, torch.dtype):
        honey_dtype = torch_dtype_to_string(dtype)
    else:
        honey_dtype = dtype
    sample = Tensor(
        [
            batch_size,
            height,
            width,
            config["in_channels"],
        ],
        name="sample",
        is_input=True,
        dtype=honey_dtype,
    )
    seq_len = IntVar([1, 77 * extend_seq_len_by_multiple])
    encoder_hidden_states = Tensor(
        [batch_size, seq_len, config["cross_attention_dim"]],
        name="encoder_hidden_states",
        is_input=True,
        dtype=honey_dtype,
    )
    timestep = Tensor(
        [batch_size],
        name="timestep",
        is_input=True,
        dtype=honey_dtype,
    )
    added_cond_kwargs = {}
    if num_add_time_ids is not None:
        add_time_ids = Tensor(
            [batch_size, 6], name="add_time_ids", is_input=True
        )
        add_text_embeds = Tensor(
            [
                batch_size,
                config["projection_class_embeddings_input_dim"]
                - (6 * config["addition_time_embed_dim"]),
            ],
            name="add_text_embeds",
            is_input=True,
        )
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
    Y = honey_module.forward(
        sample=sample,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        added_cond_kwargs=added_cond_kwargs,
    ).sample
    Y = mark_output(Y, "Y")
    # NOTE: Limited to 2gb stored constants on Windows.
    # Constants can also be applied at runtime with no limitation.
    if store_constants_in_module:
        pt_module = pt_cls.from_pretrained(hf_hub, **kwargs)
        constants = map_unet2d_condition(
            pt_module=pt_module,
            dtype=dtype,
            device=device,
            skip_keys=(),
        )
    else:
        constants = None
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
        if not store_constants_in_module:
            print("`benchmark_after_compile` requires `store_constants_in_module`.")
        else:
            benchmark_module(module=module, count=50, repeat=3)
    return module
