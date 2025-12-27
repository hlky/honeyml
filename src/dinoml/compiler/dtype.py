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
dtype definitions and utility functions of DinoML
"""

import torch

_DTYPE2BYTE = {
    "bool": 1,
    "float16": 2,
    "float32": 4,
    "float": 4,
    "int": 4,
    "int32": 4,
    "int64": 8,
    "bfloat16": 2,
    "float8_e4m3": 1,
    "float8_e5m2": 1,
}


# Maps dtype strings to DinoMLDtype enum in model_interface.h.
# Must be kept in sync!
# We can consider defining an DinoMLDtype enum to use on the Python
# side at some point, but stick to strings for now to keep things consistent
# with other Python APIs.
_DTYPE_TO_ENUM = {
    "float16": 1,
    "float32": 2,
    "float": 2,
    "int": 3,
    "int32": 3,
    "int64": 4,
    "bool": 5,
    "bfloat16": 6,
    "float8_e4m3": 7,
    "float8_e5m2": 8,
}

_ENUM_TO_DTYPE = {v: k for k, v in _DTYPE_TO_ENUM.items()}

_ENUM_TO_TORCH_DTYPE = {
    1: torch.float16,
    2: torch.float32,
    3: torch.int32,
    4: torch.int64,
    5: torch.bool,
    6: torch.bfloat16,
    7: torch.float8_e4m3fn,
    8: torch.float8_e5m2,
}


_DTYPE_TO_TORCH_DTYPE = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float": torch.float32,
    "int": torch.int32,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
    "bfloat16": torch.bfloat16,
    "float8_e4m3": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
}


def get_dtype_size(dtype: str) -> int:
    """Returns size (in bytes) of the given dtype str.

    Parameters
    ----------
    dtype: str
        A data type string.

    Returns
    ----------
    int
        Size (in bytes) of this dtype.
    """

    if dtype not in _DTYPE2BYTE:
        raise KeyError(f"Unknown dtype: {dtype}. Expected one of {_DTYPE2BYTE.keys()}")
    return _DTYPE2BYTE[dtype]


def normalize_dtype(dtype: str) -> str:
    """Returns a normalized dtype str.

    Parameters
    ----------
    dtype: str
        A data type string.

    Returns
    ----------
    str
        normalized dtype str.
    """
    if dtype == "int":
        return "int32"
    if dtype == "float":
        return "float32"
    if dtype == "float8_e4m3fn":
        return "float8_e4m3"
    return dtype


def dtype_str_to_enum(dtype: str) -> int:
    """Returns the DinoMLDtype enum value (defined in model_interface.h) of
    the given dtype str.

    Parameters
    ----------
    dtype: str
        A data type string.

    Returns
    ----------
    int
        the DinoMLDtype enum value.
    """
    if dtype not in _DTYPE_TO_ENUM:
        raise ValueError(
            f"Got unsupported input dtype {dtype}! Supported dtypes are: {list(_DTYPE_TO_ENUM.keys())}"
        )
    return _DTYPE_TO_ENUM[dtype]


def dtype_to_enumerator(dtype: str) -> str:
    """Returns the string representation of the DinoMLDtype enum
    (defined in model_interface.h) for the given dtype str.

    Parameters
    ----------
    dtype: str
        A data type string.

    Returns
    ----------
    str
        the DinoMLDtype enum string representation.
    """

    def _impl(dtype):
        if dtype == "float16":
            return "kHalf"
        elif dtype == "float32" or dtype == "float":
            return "kFloat"
        elif dtype == "int32" or dtype == "int":
            return "kInt"
        elif dtype == "int64":
            return "kLong"
        elif dtype == "bool":
            return "kBool"
        elif dtype == "bfloat16":
            return "kBFloat16"
        elif dtype == "float8_e4m3":
            return "kFloat8_e4m3"
        elif dtype == "float8_e5m2":
            return "kFloat8_e5m2"
        else:
            raise AssertionError(f"unknown dtype {dtype}")

    return f"DinoMLDtype::{_impl(dtype)}"


def is_same_dtype(dtype1: str, dtype2: str) -> bool:
    """Returns True if dtype1 and dtype2 are the same dtype and False otherwise.

    Parameters
    ----------
    dtype1: str
        A data type string.
    dtype2: str
        A data type string.

    Returns
    ----------
    bool
        whether dtype1 and dtype2 are the same dtype
    """
    return normalize_dtype(dtype1) == normalize_dtype(dtype2)
