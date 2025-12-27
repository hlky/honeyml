import json

from typing import Optional

import diffusers.models.autoencoders
import diffusers.models.transformers
import diffusers.models.unets
import transformers.models
import requests

from dinoml.frontend import Tensor
import dinoml.modeling.diffusers
import dinoml.modeling.transformers

from diffusers.models.model_loading_utils import _CLASS_REMAPPING_DICT
from huggingface_hub.utils import build_hf_headers

import dinoml.modeling.other
import dinoml.modeling.other.esrgan
import dinoml.modeling.other.esrgan.esrgan
import dinoml.modeling.other.esrgan.esrgan_pt


def mark_output(tensor: Tensor, name: str):
    tensor._attrs["is_output"] = True
    tensor._attrs["name"] = name
    shape = [d._attrs["values"] for d in tensor._attrs["shape"]]
    print(f"DinoML output `{name}` shape {shape}")
    return tensor


_CLASS_MAPPING = {
    "DiTTransformer2DModel": {
        "dinoml": dinoml.modeling.diffusers.transformers.DiTTransformer2DModel,
        "pt": diffusers.models.transformers.DiTTransformer2DModel,
    },
    "DualTransformer2DModel": {
        "dinoml": dinoml.modeling.diffusers.transformers.DualTransformer2DModel,
        "pt": diffusers.models.transformers.DualTransformer2DModel,
    },
    "HunyuanDiT2DModel": {
        "dinoml": dinoml.modeling.diffusers.transformers.HunyuanDiT2DModel,
        "pt": diffusers.models.transformers.HunyuanDiT2DModel,
    },
    "PixArtTransformer2DModel": {
        "dinoml": dinoml.modeling.diffusers.transformers.PixArtTransformer2DModel,
        "pt": diffusers.models.transformers.PixArtTransformer2DModel,
    },
    "PriorTransformer": {
        "dinoml": dinoml.modeling.diffusers.transformers.PriorTransformer,
        "pt": diffusers.models.transformers.PriorTransformer,
    },
    "Transformer2DModel": {
        "dinoml": dinoml.modeling.diffusers.transformers.Transformer2DModel,
        "pt": diffusers.models.transformers.Transformer2DModel,
    },
    "SD3Transformer2DModel": {
        "dinoml": dinoml.modeling.diffusers.transformers.SD3Transformer2DModel,
        "pt": diffusers.models.transformers.SD3Transformer2DModel,
    },
    "TransformerTemporalModel": {
        "dinoml": dinoml.modeling.diffusers.transformers.TransformerTemporalModel,
        "pt": diffusers.models.transformers.TransformerTemporalModel,
    },
    "UNet1DModel": {
        "dinoml": dinoml.modeling.diffusers.unets.UNet1DModel,
        "pt": diffusers.models.unets.UNet1DModel,
    },
    "UNet2DModel": {
        "dinoml": dinoml.modeling.diffusers.unets.UNet2DModel,
        "pt": diffusers.models.unets.UNet2DModel,
    },
    "UNet2DConditionModel": {
        "dinoml": dinoml.modeling.diffusers.unets.UNet2DConditionModel,
        "pt": diffusers.models.unets.UNet2DConditionModel,
    },
    "UNet3DConditionModel": {
        "dinoml": dinoml.modeling.diffusers.unets.UNet3DConditionModel,
        "pt": diffusers.models.unets.UNet3DConditionModel,
    },
    "I2VGenXLUNet": {
        "dinoml": dinoml.modeling.diffusers.unets.I2VGenXLUNet,
        "pt": diffusers.models.unets.I2VGenXLUNet,
    },
    "Kandinsky3UNet": {
        "dinoml": dinoml.modeling.diffusers.unets.Kandinsky3UNet,
        "pt": diffusers.models.unets.Kandinsky3UNet,
    },
    "UNetMotionModel": {
        "dinoml": dinoml.modeling.diffusers.unets.UNetMotionModel,
        "pt": diffusers.models.unets.UNetMotionModel,
    },
    "UNetSpatioTemporalConditionModel": {
        "dinoml": dinoml.modeling.diffusers.unets.UNetSpatioTemporalConditionModel,
        "pt": diffusers.models.unets.UNetSpatioTemporalConditionModel,
    },
    "StableCascadeUNet": {
        "dinoml": dinoml.modeling.diffusers.unets.StableCascadeUNet,
        "pt": diffusers.models.unets.StableCascadeUNet,
    },
    "UVit2DModel": {
        "dinoml": dinoml.modeling.diffusers.unets.UVit2DModel,
        "pt": diffusers.models.unets.UVit2DModel,
    },
    "FluxTransformer2DModel": {
        "dinoml": dinoml.modeling.diffusers.transformers.FluxTransformer2DModel,
        "pt": diffusers.models.transformers.transformer_flux.FluxTransformer2DModel,
    },
    "AutoencoderKL": {
        "dinoml": dinoml.modeling.diffusers.autoencoders.AutoencoderKL,
        "pt": diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL,
    },
    "ESRGAN": {
        "dinoml": dinoml.modeling.other.esrgan.esrgan.ESRGAN,
        "pt": dinoml.modeling.other.esrgan.esrgan_pt.RRDBNet,
    },
    "T5EncoderModel": {
        "dinoml": (
            dinoml.modeling.transformers.t5.T5EncoderModel,
            dinoml.modeling.transformers.t5.configuration_t5.T5Config,
        ),
        "pt": transformers.models.t5.modeling_t5.T5EncoderModel,
    },
}


def load_config(
    hf_hub: Optional[str] = None,
    subfolder: Optional[str] = None,
    config_file: Optional[str] = None,
    **kwargs,
):
    if config_file is not None:
        j = json.load(open(config_file, "r"))
    else:
        filename = "config.json"
        if subfolder:
            filename = f"{subfolder}/{filename}"
        url = f"https://huggingface.co/{hf_hub}/resolve/main/{filename}?download=true"
        r = requests.get(url, headers=build_hf_headers())
        if not r.ok:
            raise RuntimeError(
                f"{hf_hub}/{filename}: {r.status_code} - {r.content.decode()}"
            )
        try:
            j = r.json()
        except Exception as e:
            print(e)
    config = j
    if "architectures" in config:
        _class_name = config.pop("architectures")[0]
    else:
        _class_name = config.pop("_class_name", "")
    _diffusers_version = config.pop("_diffusers_version", None)
    _transformers_version = config.pop("transformers_version", None)
    _name_or_path = config.pop("_name_or_path", None)
    remapped_class = _CLASS_REMAPPING_DICT.get(_class_name, {}).get(
        config.get("norm_type", None), None
    )
    if remapped_class:
        _class_name = remapped_class
    print(_class_name)
    classes = _CLASS_MAPPING.get(_class_name, None)
    if classes:
        return config, classes["dinoml"], classes["pt"]
    return None, None, None
