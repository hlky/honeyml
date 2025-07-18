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
mark tensors which are parameters
"""
from typing import List

from honey.compiler.base import Tensor

# pylint: disable=C0103,W0613


def mark_special_views(sorted_graph: List[Tensor]):
    """
    Associate each tensor with an external tensor if any of the conditions are true:
    1. The tensor is a view-of-a-view of an external tensor.
    2. The tensor is a view of an input, constant or output tensor (i.e. external tensor).

    Parameters
    ----------
    sorted_graph : List[Tensor]
        The graph to mutate.
    """
    for node in sorted_graph:
        view = node._attrs["is_view_of"]
        if view is None:
            continue

        view_orig = view._attrs["external_tensor"]
        if view_orig is not None:
            node._attrs["external_tensor"] = view_orig
            continue

        view_is_input_or_constant = not view.src_ops()
        view_is_output = view._attrs["is_output"]
        if view_is_input_or_constant or view_is_output:
            node._attrs["external_tensor"] = view


def mark_param_tensor(sorted_graph: List[Tensor]):
    """
    Mark constant tensors: those that have no ops
    *and* are not explicitly marked as inputs.

    Parameters
    ----------
    sorted_graph : List[Tensor]
        The graph to mutate.
    """

    for node in reversed(sorted_graph):
        if not node.src_ops() and not node._attrs["is_input"]:
            node._attrs["is_param"] = True

        view = node._attrs["is_view_of"]
        if view is not None:
            view._attrs["has_output_aliases"] = (
                node._attrs["is_output"] or node._attrs["has_output_aliases"]
            )
