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
Debug settings
"""

from dataclasses import dataclass
from typing import Optional

from honey.utils import environ


@dataclass
class HoneyDebugSettings:
    """
    This class contains the options for configuring debug settings
    Arguments:
    check_all_nan_and_inf : bool (default: False)
        Whether or not to check this tensor is nan or inf during runtime.
    check_all_outputs : bool (default: False)
        Whether or not to print this tensor's value out during runtime.
    elements_to_check: int, optional
        Number of elements to check.
    gen_profiler_annotation : bool (default: False)
        Whether or not to add profile annotation primitives when doing codegen.
        (e.g. NVTX for CUDA and rocTX for AMD) Currently only supports NVIDIA.
    dump_honey_to_py: str, optional
        The path where the Honey graph is dumped into a .py file.
    gen_standalone : bool (default: False)
        Generate a standalone executable for the model
    """

    check_all_nan_and_inf: bool = False
    check_all_outputs: bool = False
    elements_to_check: Optional[int] = None
    gen_profiler_annotation: bool = False
    dump_honey_to_py: Optional[str] = None
    gen_standalone: bool = environ.enable_standalone_exe_generation()
