import json

from typing import Optional

import diffusers.models.autoencoders
import diffusers.models.transformers
import diffusers.models.unets
import requests

from honey.frontend import Tensor
import honey.modeling.diffusers

from diffusers.models.model_loading_utils import _CLASS_REMAPPING_DICT
from huggingface_hub.utils import build_hf_headers

import honey.modeling.other
import honey.modeling.other.esrgan
import honey.modeling.other.esrgan.esrgan
import honey.modeling.other.esrgan.esrgan_pt

def mark_output(tensor: Tensor, name: str):
    tensor._attrs["is_output"] = True
    tensor._attrs["name"] = name
    shape = [d._attrs["values"] for d in tensor._attrs["shape"]]
    print(f"Honey output `{name}` shape {shape}")
    return tensor


_CLASS_MAPPING = {
    "DiTTransformer2DModel": {
        "honey": honey.modeling.diffusers.transformers.DiTTransformer2DModel,
        "pt": diffusers.models.transformers.DiTTransformer2DModel,
    },
    "DualTransformer2DModel": {
        "honey": honey.modeling.diffusers.transformers.DualTransformer2DModel,
        "pt": diffusers.models.transformers.DualTransformer2DModel,
    },
    "HunyuanDiT2DModel": {
        "honey": honey.modeling.diffusers.transformers.HunyuanDiT2DModel,
        "pt": diffusers.models.transformers.HunyuanDiT2DModel,
    },
    "PixArtTransformer2DModel": {
        "honey": honey.modeling.diffusers.transformers.PixArtTransformer2DModel,
        "pt": diffusers.models.transformers.PixArtTransformer2DModel,
    },
    "PriorTransformer": {
        "honey": honey.modeling.diffusers.transformers.PriorTransformer,
        "pt": diffusers.models.transformers.PriorTransformer,
    },
    "Transformer2DModel": {
        "honey": honey.modeling.diffusers.transformers.Transformer2DModel,
        "pt": diffusers.models.transformers.Transformer2DModel,
    },
    "SD3Transformer2DModel": {
        "honey": honey.modeling.diffusers.transformers.SD3Transformer2DModel,
        "pt": diffusers.models.transformers.SD3Transformer2DModel,
    },
    "TransformerTemporalModel": {
        "honey": honey.modeling.diffusers.transformers.TransformerTemporalModel,
        "pt": diffusers.models.transformers.TransformerTemporalModel,
    },
    "UNet1DModel": {
        "honey": honey.modeling.diffusers.unets.UNet1DModel,
        "pt": diffusers.models.unets.UNet1DModel,
    },
    "UNet2DModel": {
        "honey": honey.modeling.diffusers.unets.UNet2DModel,
        "pt": diffusers.models.unets.UNet2DModel,
    },
    "UNet2DConditionModel": {
        "honey": honey.modeling.diffusers.unets.UNet2DConditionModel,
        "pt": diffusers.models.unets.UNet2DConditionModel,
    },
    "UNet3DConditionModel": {
        "honey": honey.modeling.diffusers.unets.UNet3DConditionModel,
        "pt": diffusers.models.unets.UNet3DConditionModel,
    },
    "I2VGenXLUNet": {
        "honey": honey.modeling.diffusers.unets.I2VGenXLUNet,
        "pt": diffusers.models.unets.I2VGenXLUNet,
    },
    "Kandinsky3UNet": {
        "honey": honey.modeling.diffusers.unets.Kandinsky3UNet,
        "pt": diffusers.models.unets.Kandinsky3UNet,
    },
    "UNetMotionModel": {
        "honey": honey.modeling.diffusers.unets.UNetMotionModel,
        "pt": diffusers.models.unets.UNetMotionModel,
    },
    "UNetSpatioTemporalConditionModel": {
        "honey": honey.modeling.diffusers.unets.UNetSpatioTemporalConditionModel,
        "pt": diffusers.models.unets.UNetSpatioTemporalConditionModel,
    },
    "StableCascadeUNet": {
        "honey": honey.modeling.diffusers.unets.StableCascadeUNet,
        "pt": diffusers.models.unets.StableCascadeUNet,
    },
    "UVit2DModel": {
        "honey": honey.modeling.diffusers.unets.UVit2DModel,
        "pt": diffusers.models.unets.UVit2DModel,
    },
    "FluxTransformer2DModel": {
        "honey": honey.modeling.diffusers.transformers.FluxTransformer2DModel,
        "pt": diffusers.models.transformers.transformer_flux.FluxTransformer2DModel,
    },
    "AutoencoderKL": {
        "honey": honey.modeling.diffusers.autoencoders.AutoencoderKL,
        "pt": diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL,
    },
    "ESRGAN": {
        "honey": honey.modeling.other.esrgan.esrgan.ESRGAN,
        "pt": honey.modeling.other.esrgan.esrgan_pt.RRDBNet,
    }
}


def load_config(
    hf_hub: Optional[str] = None,
    subfolder: Optional[str] = None,
    config_file: Optional[str] = None,
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
            raise RuntimeError(f"{hf_hub}/{filename}: {r.status_code} - {r.content.decode()}")
        try:
            j = r.json()
        except Exception as e:
            print(e)
    config = j
    _class_name = config.pop("_class_name", "")
    _diffusers_version = config.pop("_diffusers_version")
    _name_or_path = config.pop("_name_or_path", None)
    remapped_class = _CLASS_REMAPPING_DICT.get(_class_name, {}).get(
        config.get("norm_type", None), None
    )
    if remapped_class:
        _class_name = remapped_class
    print(_class_name)
    classes = _CLASS_MAPPING.get(_class_name, None)
    if classes:
        return config, classes["honey"], classes["pt"]
    return None, None, None
