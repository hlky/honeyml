#  Copyright 2025 hlky. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""
CUDA tensor ops module init
"""
from honey.backend.cuda.tensor import (
    arange,
    argmax,
    batch_gather,
    cast,
    concatenate,
    concatenate_tanh,
    dynamic_slice,
    expand,
    full,
    gather,
    identity,
    index_select,
    jagged_to_padded_dense,
    masked_select,
    meshgrid,
    pad,
    padded_dense_to_jagged,
    permute,
    permute021,
    permute0213,
    permute102,
    permute210,
    pixel_shuffle,
    pixel_unshuffle,
    relational,
    repeat_interleave,
    slice_reshape_scatter,
    slice_scatter,
    split,
    topk,
    t5_layer_norm,
    where,
)

__all__ = [
    "arange",
    "argmax",
    "batch_gather",
    "cast",
    "concatenate",
    "concatenate_tanh",
    "dynamic_slice",
    "expand",
    "full",
    "gather",
    "relational",
    "identity",
    "jagged_to_padded_dense",
    "index_select",
    "masked_select",
    "meshgrid",
    "pad",
    "padded_dense_to_jagged",
    "permute",
    "permute021",
    "permute0213",
    "permute102",
    "permute210",
    "pixel_shuffle",
    "pixel_unshuffle",
    "repeat_interleave",
    "slice_reshape_scatter",
    "slice_scatter",
    "split",
    "topk",
    "t5_layer_norm",
    "where",
]
