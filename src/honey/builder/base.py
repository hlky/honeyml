from typing import Any, Dict, List, Optional, cast, Tuple, Union, Callable

import torch

from honey.compiler import compile_model
from honey.frontend import IntImm, IntVar, Tensor, nn
from honey.mapping.autoencoder_kl import map_autoencoder_kl
from honey.mapping.unet2d_condition import map_unet2d_condition
from honey.testing import detect_target
from honey.testing.benchmark_honey import benchmark_module
from honey.utils.build_utils import (
    build_tensors_from_annotations,
    get_device_name,
    get_sm,
)
from honey.utils.torch_utils import torch_dtype_to_string

from honey.builder.config import load_config, mark_output


class Build:
    model_name: str = "model.{label}.{device_name}.sm{sm}"
    model_type: Optional[str]

    model_forward: str = "forward"
    model_output: str = "sample"
    model_output_names: List[str]

    map_function: Callable
    map_function_skip_keys: Tuple = ()

    constants: Optional[Dict[str, torch.Tensor]]
    honey_dtype: str

    def __init__(
        self,
        hf_hub: str,
        label: str,
        dtype: Union[str, torch.dtype],
        device: Union[str, torch.device],
        build_kwargs: Dict[str, Any],
        model_kwargs: Dict[str, Any],
        benchmark_after_compile: bool = True,
        store_constants_in_module: bool = True,
    ):
        self.hf_hub = hf_hub
        self.label = label
        self.dtype = dtype
        self.device = device
        self.build_kwargs = build_kwargs
        self.model_kwargs = model_kwargs
        self.benchmark_after_compile = benchmark_after_compile
        self.store_constants_in_module = store_constants_in_module

    def _model_name(self):
        return {}

    def create_model_name(self):
        device_name = get_device_name()
        sm = get_sm()
        model_name_args = {
            "label": self.label,
            "device_name": device_name,
            "sm": sm,
        }
        if self.model_type is not None:
            model_name_args["model_type"] = self.model_type
        model_name_args.update(self._model_name())
        self.model_name = self.model_name.format(**model_name_args)

    def set_honey_dtype(self):
        if isinstance(self.dtype, torch.dtype):
            self.honey_dtype = torch_dtype_to_string(self.dtype)
        else:
            self.honey_dtype = self.dtype

    def create_module(self):
        self.config, self.honey_cls, self.pt_cls = load_config(
            self.hf_hub, **self.model_kwargs
        )
        self.honey_module = cast(
            nn.Module, self.honey_cls(**self.config, dtype=self.honey_dtype)
        )
        self.honey_module.name_parameter_tensor()

    def create_input_tensors(self):
        self.input_tensors = build_tensors_from_annotations(
            getattr(self.honey_module, self.model_forward),
            symbolic_values=self.build_kwargs,
            config=self.config,
        )

    def create_output_tensors(self):
        output_tensors: Union[Tensor, List[Tensor]] = getattr(
            getattr(self.honey_module, self.model_forward)(**self.input_tensors),
            self.model_output,
        )
        if isinstance(output_tensors, Tensor):
            output_tensors = [output_tensors]
        for idx, output_tensor in enumerate(output_tensors):
            output_tensors[idx] = mark_output(
                output_tensor, self.model_output_names[idx]
            )
        self.output_tensors = output_tensors

    def create_constants(self):
        if self.store_constants_in_module:
            pt_module = self.pt_cls.from_pretrained(self.hf_hub, **self.model_kwargs)
            self.constants = self.map_function(
                pt_module=pt_module,
                dtype=self.dtype,
                device=self.device,
                skip_keys=self.map_function_skip_keys,
            )

    def compile(self):
        target = detect_target()
        module = compile_model(
            self.output_tensors,
            target,
            "./tmp",
            self.model_name,
            constants=self.constants,
            dll_name=f"{self.model_name}.so",
        )
        if self.benchmark_after_compile:
            if not self.store_constants_in_module:
                print("`benchmark_after_compile` requires `store_constants_in_module`.")
            else:
                benchmark_module(module=module, count=50, repeat=3)
        return module

    def __call__(self):
        self.create_model_name()
        self.set_honey_dtype()
        self.create_module()
        self.create_input_tensors()
        self.create_output_tensors()
        self.create_constants()
        return self.compile()


class AutoencoderKLDecodeBuilder(Build):
    model_name = "autoencoder_kl.{model_type}.{label}.{resolution}.{device_name}.sm{sm}"
    model_type = "decode"
    map_function = map_autoencoder_kl
    map_function_skip_keys = (
        "quant.",
        "encoder.",
    )
    model_forward = "_decode"
    model_output_names = ["Y"]

    def _model_name(self):
        if "resolution" not in self.build_kwargs:
            raise ValueError("Expected `resolution` in `build_kwargs`.")
        resolution = self.build_kwargs.pop("resolution")
        if isinstance(resolution, tuple):
            resolution_label = resolution[-1]
        else:
            resolution_label = resolution
        self.build_kwargs["height"] = resolution
        self.build_kwargs["width"] = resolution
        return {"resolution": resolution_label}


def _model_name_with_resolution(self):
    if "resolution" not in self.build_kwargs:
        raise ValueError("Expected `resolution` in `build_kwargs`.")
    resolution = self.build_kwargs.pop("resolution")
    if isinstance(resolution, tuple):
        resolution_label = resolution[-1]
    else:
        resolution_label = resolution
    self.build_kwargs["height"] = resolution
    self.build_kwargs["width"] = resolution
    return {"resolution": resolution_label}
