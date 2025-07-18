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
Slice scatter CUDA implementation.
"""

from honey.backend import registry
from honey.backend.backend_spec import CUDASpec
from honey.backend.common.tensor import slice_common


@registry.reg("cuda.slice_scatter.func_decl")
def gen_function_decl(func_attrs):
    """Generate function declaration.

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        Stores the operation attributes.
    Returns
    -------
    str
        Rendered function declaration.
    """
    return slice_common.gen_function_decl(func_attrs, CUDASpec())


@registry.reg("cuda.slice_scatter.gen_function")
def gen_function(func_attrs):
    """Generates function body.

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        Stores the operation attributes.

    Returns
    -------
    str
        Rendered function body.
    """
    # TODO: consider to profile elems_per_thread
    elems_per_thread = 8 if len(func_attrs["inputs"]) == 1 else 256
    output_accessor = func_attrs["output_accessors"][0]
    output_offset = output_accessor.offset
    return slice_common.gen_function(
        func_attrs,
        backend_spec=CUDASpec(),
        elems_per_thread=elems_per_thread,
        output_offset=output_offset,
        update_output_shape=False,
    )


@registry.reg("cuda.slice_scatter.func_call")
def gen_function_call(func_attrs, indent="  "):
    """Generates function call.

    Parameters
    ----------
    func_attrs : Dict[str, Any]
        Stores the operation attributes.
    indent : str, optional
        Indent for template, by default "  ".

    Returns
    -------
    str
        Rendered function call.
    """
    slice_ops = func_attrs["slice_ops"]
    assert len(slice_ops) >= 1
    start_indices = [op._attrs["start_indices"] for op in slice_ops]
    end_indices = [op._attrs["end_indices"] for op in slice_ops]
    return slice_common.gen_function_call(
        backend_spec=CUDASpec(),
        func_name=func_attrs["name"],
        inputs=func_attrs["inputs"],
        outputs=func_attrs["outputs"],
        start_indices=start_indices,
        end_indices=end_indices,
        dim=func_attrs["scatter_dim"],
        indent=indent,
    )
