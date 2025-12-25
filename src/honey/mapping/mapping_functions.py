from typing import Callable, Dict, Iterable, Union

import torch
from honey.utils.torch_utils import string_to_torch_dtype


def conv2d_permute(key: str, tensor: torch.Tensor):
    if (
        tensor.ndim == 4
        and "conv" in key
        and "norm" not in key
        and key.endswith("_weight")
    ):
        return torch.permute(tensor, [0, 2, 3, 1]).contiguous()
    else:
        return tensor


def conv2d_pad(key: str, tensor: torch.Tensor, pad_to_multiple_of: int = 4):
    if (
        tensor.ndim == 4
        and "conv" in key
        and "norm" not in key
        and key.endswith("_weight")
        and tensor.shape[-1] % 4 != 0
    ):
        channels = tensor.shape[-1]
        pad_by = pad_to_multiple_of - (channels % pad_to_multiple_of)
        pad = (0, pad_by, 0, 0, 0, 0, 0, 0)
        print(f"Padding {key=}, {channels=}, {pad_by=}.")
        return torch.nn.functional.pad(tensor, pad=pad)
    else:
        return tensor


def _map(
    pt_module: Union[torch.nn.Module, Dict[str, torch.Tensor]],
    dtype: Union[str, torch.dtype],
    device: Union[str, torch.device],
    skip_keys: Iterable[str],
    mapping_fn: Iterable[Callable],
):
    if isinstance(pt_module, torch.nn.Module):
        pt_params = dict(pt_module.named_parameters())
    elif isinstance(pt_module, dict):
        pt_params = pt_module
    else:
        raise ValueError("Expected `torch.nn.Module` or `Dict[str, torch.Tensor]`.")
    if isinstance(dtype, str):
        dtype = string_to_torch_dtype(dtype)
    if isinstance(device, str):
        device = torch.device(device)
    honey_params = {}
    for key, tensor in pt_params.items():
        if any(key.startswith(k) for k in skip_keys):
            continue
        tensor_dtype = tensor.dtype
        if tensor_dtype.is_floating_point and tensor_dtype != dtype:
            tensor = tensor.to(dtype)
        key = key.replace(".", "_")
        for map_fn in mapping_fn:
            tensor = map_fn(key, tensor)
        if tensor.device != device:
            tensor = tensor.to(device)
        honey_params[key] = tensor
    return honey_params
