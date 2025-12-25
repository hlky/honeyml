from .mapping_functions import _map, conv2d_permute, conv2d_pad
from typing import Callable, Dict, Iterable, Union

import torch


def map_transformer_flux(
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
