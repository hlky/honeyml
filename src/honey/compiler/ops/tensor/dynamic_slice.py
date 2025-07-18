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
Dynamic_slice.
"""
from typing import List, Optional, Union

import sympy

from honey import backend
from honey.backend import registry
from honey.compiler.base import IntImm, IntVar, IntVarTensor, Operator, Tensor
from honey.utils import shape_utils

# pylint: disable=C0103,W0221

# FIXME: We use MAX_INT32 to represent the end position in a sliced
# dimension for now, because we use int32_t to represent indices in
# the generated backend CUDA/C++ code. After we replace int32_t with
# int64_t in ourbackend, we will also need to replace MAX_INT32 with
# MAX_INT64.
MAX_INT32 = pow(2, 31) - 1


class dynamic_slice(Operator):
    """
    Cut the source tensor into slices specified by a list of start indices and a list of end indices.

    Args:
        x (Tensor): input tensor
        start_indices (List[int]) : similar to PyTorch and numpy, indices can be negative
        end_indices (List[int]) : end_index is not included. Similar to PyTorch and numpy, indices can be negative.

    Returns:
        List[Tensor] : the list of sliced tensors.
    """

    def __init__(self) -> None:
        super().__init__()
        self._attrs["op"] = "dynamic_slice"
        self._attrs["has_profiler"] = False

    @staticmethod
    def normalize_start_end_indices(dim_val: int, start: int, end: int) -> List[int]:
        """
        return normalized start and end indices which fall into a well-formed
        range like below:
        0 <= start <= end <= dim_val
        """
        # handle negative indices
        start = start if start >= 0 else dim_val + start
        start = 0 if start < 0 else start
        end = end if end >= 0 else dim_val + end
        end = 0 if end < 0 else end

        start = dim_val if start > dim_val else start
        end = dim_val if end > dim_val else end
        start = end if start > end else start
        return [start, end]

    def _infer_dynamic_dim(self, dim: IntVar, start_index: int, end_index: int):
        values = dim._attrs["values"]
        new_values = []

        for value in values:
            start, end = dynamic_slice.normalize_start_end_indices(
                value, start_index, end_index
            )
            new_values.append(end - start)
        new_values = sorted(set(new_values))

        start_sym = (
            start_index if start_index >= 0 else dim.symbolic_value() + start_index
        )
        end_sym = end_index if end_index >= 0 else dim.symbolic_value() + end_index

        start_sym = sympy.Min(dim.symbolic_value(), sympy.Max(0, start_sym))
        end_sym = sympy.Min(dim.symbolic_value(), sympy.Max(0, end_sym))

        symbolic_value = sympy.Max(0, end_sym - start_sym)

        return shape_utils.gen_int_var(new_values, symbolic_value=symbolic_value)

    def _infer_shapes(
        self,
        x: Tensor,
        start_indices: List[Union[IntVar, IntVarTensor, Optional[int]]],
        end_indices: List[Union[IntVar, IntVarTensor, Optional[int]]],
    ) -> List[IntVar]:
        """Infers shape for dynamic_slice."""
        # TODO: Handle start_indices/end_indices that are not int.

        x_shape = x._attrs["shape"]
        output_shape = []
        for dim_val, start, end in zip(x_shape, start_indices, end_indices):
            if start == 0 and end == MAX_INT32:
                # Slicing along the whole dim.
                output_shape.append(dim_val)
            elif isinstance(dim_val, IntImm):
                # Slicing a static dimension.
                start, end = dynamic_slice.normalize_start_end_indices(
                    dim_val.value(), start, end
                )
                output_shape.append(IntImm(end - start))
            elif start >= 0 and end >= 0:
                # Fixed size from start and end.
                output_shape.append(IntImm(end - start))
            else:
                output_shape.append(self._infer_dynamic_dim(dim_val, start, end))

        return output_shape

    def __call__(
        self,
        x: Tensor,
        start_indices: List[Union[IntVar, IntVarTensor, Optional[int]]],
        end_indices: List[Union[IntVar, IntVarTensor, Optional[int]]],
    ) -> List[Tensor]:
        """
        Parameters
        ----------
        x : Tensor
            Input tensor.
        start_indices : List[Union[IntVar, IntVarTensor, Optional[int]]]
            Similar to PyTorch and numpy, indices can be negative
        end_indices : List[Union[IntVar, IntVarTensor, Optional[int]]]
            end_index is not included. Similar to PyTorch and
                numpy, indices can be negative.

        Returns
        -------
        List[Tensor]
            Output tensors.
        """

        x_shape = x._attrs["shape"]
        if len(start_indices) != len(end_indices):
            raise RuntimeError("len(start_indices) must equal to len(end_indices)")
        rank = len(x_shape)
        if rank != len(start_indices):
            raise RuntimeError(
                "input rank expected to be equal to the length of start_indices"
                ", but got {} and {}".format(rank, len(start_indices))
            )

        start_indices = [
            shape_utils.convert_IntVar_to_int(idx) if idx is not None else 0
            for idx in start_indices
        ]
        end_indices = [
            shape_utils.convert_IntVar_to_int(idx) if idx is not None else MAX_INT32
            for idx in end_indices
        ]

        self._attrs["inputs"] = [x]
        self._attrs["start_indices"] = start_indices
        self._attrs["end_indices"] = end_indices
        self._set_depth()

        output_shape = self._infer_shapes(x, start_indices, end_indices)
        output = Tensor(output_shape, src_ops={self}, dtype=x._attrs["dtype"])
        self._attrs["outputs"] = [output]
        return output

    def _get_func(self, fmt_str):
        """
        Parameters
        ----------
        inputs : string
            format string to create func_key for looking up func
            from the registry

        Returns
        -------
        the function generator
        """
        target = backend.target.Target.current()
        func_key = fmt_str.format(target=target.name(), op=self._attrs["op"])
        return registry.get(func_key)

    def gen_function(self) -> str:
        func = self._get_func("{target}.{op}.gen_function")
        return func(self._attrs)

    def _inputs_for_pseudo_code(self):
        return self._attrs["inputs"]

    def _args_for_pseudo_code(self):
        return [
            f"start_indices=[{self._pseudo_code_helper(self._attrs['start_indices'], with_shape=True)}]",
            f"end_indices=[{self._pseudo_code_helper(self._attrs['end_indices'], with_shape=True)}]",
        ]
