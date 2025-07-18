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
Applies graph transformations.
"""
from typing import List

from honey.compiler.base import Tensor
from honey.compiler.transform.apply_padding import apply_padding
from honey.compiler.transform.dedup_make_jagged_ops import dedup_make_jagged_ops
from honey.compiler.transform.fuse_bmm_permute import fuse_bmm_permute
from honey.compiler.transform.fuse_conv_elementwise import fuse_conv_elementwise
from honey.compiler.transform.fuse_duplicate_fused_elementwise import (
    fuse_duplicate_fused_elementwise,
)
from honey.compiler.transform.fuse_expand_bmm import fuse_expand_bmm
from honey.compiler.transform.fuse_group_ops import fuse_group_ops
from honey.compiler.transform.fuse_mm_elementwise import fuse_mm_elementwise
from honey.compiler.transform.fuse_mm_reshape_permute import (
    fuse_mm_reshape_permute,
)
from honey.compiler.transform.fuse_ops import (
    fuse_elementwise,
    fuse_ops,
    process_singleton_elementwise,
)
from honey.compiler.transform.fuse_parallel_gemms import (
    fuse_parallel_gemms,
    fuse_single_source_parallel_gemms,
)
from honey.compiler.transform.fuse_permute_bmm_and_gemm import (
    fuse_permute_bmm_and_gemm,
)
from honey.compiler.transform.move_view_ops import move_view_op_before_concat
from honey.compiler.transform.remove_elementwise_no_ops import (
    remove_elementwise_no_ops,
)
from honey.compiler.transform.split_large_concat_ops import split_large_concat_ops
from honey.compiler.transform.split_large_slice_scatter_ops import (
    split_large_slice_scatter_ops,
)
from honey.compiler.transform.split_large_split_ops import split_large_split_ops
from honey.compiler.transform.transform_memory_ops import transform_memory_ops
from honey.compiler.transform.transform_merge_view_ops import merge_view_ops
from honey.compiler.transform.transform_odd_alignment import (
    transform_odd_alignment,
)
from honey.compiler.transform.transform_permutations import eliminate_permutations
from honey.compiler.transform.transform_permute_to_reshape import (
    transform_permute_to_reshape,
)
from honey.compiler.transform.transform_special_ops import transform_special_ops
from honey.compiler.transform.transform_strided_ops import transform_strided_ops

from honey.utils import graph_utils


def optimize_graph(
    sorted_graph: List[Tensor], workdir: str, optimize=True
) -> List[Tensor]:
    """Applies graph optimizations, including

    - fuse permute and bmm
    - fuse permute and gemm
    - transform odd alignment
    - fuse conv and elementwise
    - fuse gemm and elementwise
    - fuse elementwise ops
    - fuse parallel gemms
    - fuse group ops
    - transform special ops
    - transform strided ops
    - fuse bmm and permute
    - transform memory ops
    - apply padding

    Parameters
    ----------
    sorted_graph : List[Tensor]
        Input graph
    workdir : str
        working directory

    Returns
    -------
    List[Tensor]
        Fused graph
    """

    funcs = [
        remove_elementwise_no_ops,
        dedup_make_jagged_ops,
        fuse_permute_bmm_and_gemm,
        fuse_bmm_permute,
        fuse_expand_bmm,
        transform_odd_alignment,
        fuse_conv_elementwise,
        fuse_single_source_parallel_gemms,
        fuse_mm_elementwise,
        fuse_mm_reshape_permute,
        # make sure we run move_view_op_before_concat before transform_memory_ops
        move_view_op_before_concat,
        merge_view_ops,
        transform_memory_ops,
        fuse_ops,
        fuse_elementwise,
        # need to run before transform_strided_ops to fuse strided ops + concat
        # and transform_memory_ops to fuse split + concat
        fuse_parallel_gemms,
        fuse_group_ops,
        # This needs to be run after fuse_ops() to avoid handling elementwise
        # op directly. After fuse_ops, there are only FusedElementwise ops.
        transform_special_ops,
        apply_padding,
        # apply_padding may introduce new concats that can be fused
        move_view_op_before_concat,
        transform_memory_ops,
        transform_strided_ops,
        split_large_slice_scatter_ops,
        split_large_concat_ops,
        split_large_split_ops,
        transform_permute_to_reshape,
        transform_memory_ops,
        eliminate_permutations,
        # fuse_duplicate_fused_elementwise must run after elementwise fusion and
        # after passes that modify/replace a fused_elementwise's input/output accessor.
        fuse_duplicate_fused_elementwise,
    ]

    if not optimize:
        # 1 - Convert elementwise ops to singleton fused_elementwise ops
        # 2 - Padding also needs to be done for the model to be executable.
        funcs = [
            process_singleton_elementwise,
            apply_padding,
            split_large_slice_scatter_ops,
            split_large_concat_ops,
            split_large_split_ops,
        ]

    for i, func in enumerate(funcs):
        sorted_graph = func(sorted_graph, workdir)
        graph_utils.dump_graph_debug_str_to_file(
            sorted_graph, workdir, f"{i:02}-{func.__name__}"
        )

    return sorted_graph
