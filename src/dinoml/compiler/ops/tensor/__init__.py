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
# flake8: noqa
"""
reduce module init
"""

from dinoml.compiler.ops.tensor.arange import arange
from dinoml.compiler.ops.tensor.argmax import argmax
from dinoml.compiler.ops.tensor.batch_gather import batch_gather
from dinoml.compiler.ops.tensor.cast import cast
from dinoml.compiler.ops.tensor.chunk import chunk
from dinoml.compiler.ops.tensor.concatenate import concatenate
from dinoml.compiler.ops.tensor.concatenate_tanh import concatenate_tanh
from dinoml.compiler.ops.tensor.dynamic_slice import dynamic_slice
from dinoml.compiler.ops.tensor.expand import expand
from dinoml.compiler.ops.tensor.full import full
from dinoml.compiler.ops.tensor.gather import gather
from dinoml.compiler.ops.tensor.gelu_new import gelu_new
from dinoml.compiler.ops.tensor.get_3d_sincos_pos_embed import get_3d_sincos_pos_embed
from dinoml.compiler.ops.tensor.get_2d_sincos_pos_embed import get_2d_sincos_pos_embed
from dinoml.compiler.ops.tensor.identity import identity
from dinoml.compiler.ops.tensor.index_select import index_select
from dinoml.compiler.ops.tensor.jagged_to_padded_dense import jagged_to_padded_dense
from dinoml.compiler.ops.tensor.masked_select import masked_select
from dinoml.compiler.ops.tensor.meshgrid import meshgrid
from dinoml.compiler.ops.tensor.pad import pad
from dinoml.compiler.ops.tensor.padded_dense_to_jagged import padded_dense_to_jagged
from dinoml.compiler.ops.tensor.permute import permute
from dinoml.compiler.ops.tensor.permute021 import permute021
from dinoml.compiler.ops.tensor.permute0213 import permute0213
from dinoml.compiler.ops.tensor.permute102 import permute102
from dinoml.compiler.ops.tensor.permute210 import permute210
from dinoml.compiler.ops.tensor.pixel_shuffle import pixel_shuffle
from dinoml.compiler.ops.tensor.pixel_unshuffle import pixel_unshuffle
from dinoml.compiler.ops.tensor.randn import randn
from dinoml.compiler.ops.tensor.relational import eq, ge, gt, le, lt, ne
from dinoml.compiler.ops.tensor.relative_attention_bias import relative_attention_bias
from dinoml.compiler.ops.tensor.repeat_interleave import repeat_interleave
from dinoml.compiler.ops.tensor.size import size
from dinoml.compiler.ops.tensor.slice_reshape_scatter import slice_reshape_scatter
from dinoml.compiler.ops.tensor.slice_scatter import slice_scatter
from dinoml.compiler.ops.tensor.split import split
from dinoml.compiler.ops.tensor.stack import stack
from dinoml.compiler.ops.tensor.topk import topk
from dinoml.compiler.ops.tensor.transpose import transpose
from dinoml.compiler.ops.tensor.t5_layer_norm import t5_layer_norm
from dinoml.compiler.ops.tensor.where import where
