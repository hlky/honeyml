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
from dinoml.compiler.transform.bind_constants import bind_constants
from dinoml.compiler.transform.constant_folding import constant_folding
from dinoml.compiler.transform.fuse_conv_elementwise import fuse_conv_elementwise
from dinoml.compiler.transform.fuse_expand_bmm import fuse_expand_bmm
from dinoml.compiler.transform.fuse_group_ops import (
    fuse_group_gemm_ops,
    fuse_group_layernorm_ops,
    fuse_group_ops,
)
from dinoml.compiler.transform.fuse_mm_elementwise import fuse_mm_elementwise
from dinoml.compiler.transform.fuse_ops import fuse_ops
from dinoml.compiler.transform.fuse_permute_bmm_and_gemm import (
    fuse_permute_bmm_and_gemm,
)
from dinoml.compiler.transform.mark_param_tensor import (
    mark_param_tensor,
    mark_special_views,
)
from dinoml.compiler.transform.memory_planning import memory_planning
from dinoml.compiler.transform.move_view_ops import move_view_op_before_concat
from dinoml.compiler.transform.name_graph import dedup_symbolic_name, name_graph
from dinoml.compiler.transform.optimize_graph import optimize_graph
from dinoml.compiler.transform.profile import profile
from dinoml.compiler.transform.refine_graph import refine_graph
from dinoml.compiler.transform.remove_no_ops import remove_no_ops
from dinoml.compiler.transform.remove_unused_ops import remove_unused_ops
from dinoml.compiler.transform.split_large_concat_ops import split_large_concat_ops
from dinoml.compiler.transform.split_large_split_ops import split_large_split_ops
from dinoml.compiler.transform.toposort import toposort
from dinoml.compiler.transform.transform_memory_ops import transform_memory_ops
from dinoml.compiler.transform.transform_merge_slice_ops import merge_slice_ops
from dinoml.compiler.transform.transform_odd_alignment import (
    transform_odd_alignment,
)
from dinoml.compiler.transform.transform_special_ops import transform_special_ops
from dinoml.compiler.transform.transform_strided_ops import transform_strided_ops
