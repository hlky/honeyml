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
CUDA concatenate function
"""

from honey.backend import registry
from honey.backend.backend_spec import CUDASpec
from honey.backend.common import split_common


@registry.reg("cuda.split.func_decl")
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
    return split_common.gen_function_decl(
        func_attrs=func_attrs, backend_spec=CUDASpec()
    )


@registry.reg("cuda.split.gen_function")
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
    return split_common.gen_function(func_attrs=func_attrs, backend_spec=CUDASpec())


@registry.reg("cuda.split.func_call")
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
    return split_common.gen_function_call(
        func_attrs=func_attrs, backend_spec=CUDASpec()
    )
