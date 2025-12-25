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

from honey.compiler.ops.tensor.arange import arange
from honey.compiler.ops.tensor.argmax import argmax
from honey.compiler.ops.tensor.batch_gather import batch_gather
from honey.compiler.ops.tensor.cast import cast
from honey.compiler.ops.tensor.chunk import chunk
from honey.compiler.ops.tensor.concatenate import concatenate
from honey.compiler.ops.tensor.concatenate_tanh import concatenate_tanh
from honey.compiler.ops.tensor.dynamic_slice import dynamic_slice
from honey.compiler.ops.tensor.expand import expand
from honey.compiler.ops.tensor.full import full
from honey.compiler.ops.tensor.gather import gather
from honey.compiler.ops.tensor.gelu_new import gelu_new
from honey.compiler.ops.tensor.identity import identity
from honey.compiler.ops.tensor.index_select import index_select
from honey.compiler.ops.tensor.jagged_to_padded_dense import jagged_to_padded_dense
from honey.compiler.ops.tensor.masked_select import masked_select
from honey.compiler.ops.tensor.meshgrid import meshgrid
from honey.compiler.ops.tensor.pad import pad
from honey.compiler.ops.tensor.padded_dense_to_jagged import padded_dense_to_jagged
from honey.compiler.ops.tensor.permute import permute
from honey.compiler.ops.tensor.permute021 import permute021
from honey.compiler.ops.tensor.permute0213 import permute0213
from honey.compiler.ops.tensor.permute102 import permute102
from honey.compiler.ops.tensor.permute210 import permute210
from honey.compiler.ops.tensor.pixel_shuffle import pixel_shuffle
from honey.compiler.ops.tensor.pixel_unshuffle import pixel_unshuffle
from honey.compiler.ops.tensor.relational import eq, ge, gt, le, lt, ne
from honey.compiler.ops.tensor.relative_attention_bias import relative_attention_bias
from honey.compiler.ops.tensor.repeat_interleave import repeat_interleave
from honey.compiler.ops.tensor.size import size
from honey.compiler.ops.tensor.slice_reshape_scatter import slice_reshape_scatter
from honey.compiler.ops.tensor.slice_scatter import slice_scatter
from honey.compiler.ops.tensor.split import split
from honey.compiler.ops.tensor.stack import stack
from honey.compiler.ops.tensor.topk import topk
from honey.compiler.ops.tensor.transpose import transpose
from honey.compiler.ops.tensor.t5_layer_norm import t5_layer_norm
from honey.compiler.ops.tensor.where import where
