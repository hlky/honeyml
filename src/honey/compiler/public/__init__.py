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
This file defines public tensor concepts and ops that are exposed to
external converters, e.g. FX2Honey.
"""

# pylint: disable=C0413,W0105

"""Shape"""
from honey.compiler.base import IntImm, IntVar

"""Tensor"""
from honey.compiler.base import IntVarTensor, Tensor

"""Profiling"""
from honey.compiler.base import DynamicProfileStrategy

"""Operators"""

"""Elementwise"""
from honey.compiler.ops.common.elementwise import clamp, elementwise
from honey.compiler.ops.common.epilogue import FuncEnum

from honey.compiler.ops.common.int_elementwise import int_elementwise

"""GEMM"""
from honey.compiler.ops.gemm_universal.bmm_xxx import bmm_rcr, bmm_rrr
from honey.compiler.ops.gemm_universal.gemm_rcr import gemm_rcr
from honey.compiler.ops.gemm_universal.gemm_rcr_bias import gemm_rcr_bias
from honey.compiler.ops.gemm_universal.gemm_rrr import gemm_rrr

"""Reduce"""
from honey.compiler.ops.reduce.reduce_max import reduce_max
from honey.compiler.ops.reduce.reduce_mean import reduce_mean
from honey.compiler.ops.reduce.reduce_min import reduce_min
from honey.compiler.ops.reduce.reduce_sum import reduce_sum
from honey.compiler.ops.reduce.var import var
from honey.compiler.ops.reduce.vector_norm import vector_norm

"""View ops"""
from honey.compiler.ops.common.view_ops import flatten, reshape, squeeze, unsqueeze

"""Functions"""
from honey.compiler.ops.conv.conv2d import conv2d
from honey.compiler.ops.conv.conv3d import conv3d
from honey.compiler.ops.conv.conv3d_bias import conv3d_bias
from honey.compiler.ops.conv.depthwise_conv3d import depthwise_conv3d
from honey.compiler.ops.conv.transposed_conv2d import transposed_conv2d
from honey.compiler.ops.conv.transposed_conv2d_bias import transposed_conv2d_bias
from honey.compiler.ops.groupnorm.groupnorm import group_norm
from honey.compiler.ops.layernorm.group_layernorm import group_layernorm
from honey.compiler.ops.layernorm.group_layernorm_sigmoid_mul import (
    group_layernorm_sigmoid_mul,
)
from honey.compiler.ops.layernorm.layernorm import layernorm
from honey.compiler.ops.padding import ndhwc3to8, nhwc3to8, pad_last_dim
from honey.compiler.ops.pool.avg_pool2d import avg_pool2d
from honey.compiler.ops.pool.max_pool2d import max_pool2d
from honey.compiler.ops.softmax.softmax import softmax
from honey.compiler.ops.tensor.index_select import index_select
from honey.compiler.ops.tensor.masked_select import masked_select
from honey.compiler.ops.tensor.size import size
from honey.compiler.ops.tensor.topk import topk

"""Memory ops"""
from honey.compiler.ops.tensor.cast import cast
from honey.compiler.ops.tensor.chunk import chunk
from honey.compiler.ops.tensor.concatenate import concatenate
from honey.compiler.ops.tensor.dynamic_slice import dynamic_slice
from honey.compiler.ops.tensor.expand import expand
from honey.compiler.ops.tensor.full import full
from honey.compiler.ops.tensor.identity import identity
from honey.compiler.ops.tensor.permute import permute
from honey.compiler.ops.tensor.split import split

"""Python ops"""
from honey.compiler.ops.common import getitem, list_construct, tuple_construct
