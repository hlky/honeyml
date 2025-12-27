from .mapping_functions import _map
from typing import Callable, Dict, Iterable, Union

import torch


def conv2d_permute(key: str, tensor: torch.Tensor):
    if tensor.ndim == 4:
        print(f"Permuting {key=}")
        return torch.permute(tensor, [0, 2, 3, 1]).contiguous()
    else:
        return tensor


def conv2d_pad(key: str, tensor: torch.Tensor, pad_to_multiple_of: int = 4):
    if tensor.ndim == 4 and tensor.shape[-1] % 4 != 0:
        channels = tensor.shape[-1]
        pad_by = pad_to_multiple_of - (channels - pad_to_multiple_of)
        pad = (0, pad_by, 0, 0, 0, 0, 0, 0)
        print(f"Padding {key=} with {pad=}")
        return torch.nn.functional.pad(tensor, pad=pad)
    else:
        return tensor


def map_unet2d_condition(
    self,
    pt_module: Union[torch.nn.Module, Dict[str, torch.Tensor]],
    dtype: Union[str, torch.dtype],
    device: Union[str, torch.device],
    skip_keys: Iterable[str],
    mapping_fn: Iterable[Callable] = (conv2d_permute, conv2d_pad),
):
    return _map(
        pt_module=pt_module,
        dtype=dtype,
        device=device,
        skip_keys=skip_keys,
        mapping_fn=mapping_fn,
    )
