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
Upsampling module init.
"""

from dinoml.compiler.ops.upsample.upsampling1d import upsampling1d
from dinoml.compiler.ops.upsample.upsampling1d_add import upsampling1d_add
from dinoml.compiler.ops.upsample.upsampling2d import upsampling2d
from dinoml.compiler.ops.upsample.upsampling2d_add import upsampling2d_add
from dinoml.compiler.ops.upsample.upsampling3d import upsampling3d
from dinoml.compiler.ops.upsample.upsampling3d_add import upsampling3d_add
from dinoml.compiler.ops.upsample.upsampling3d_compress_time import (
    upsampling3d_compress_time,
)

__all__ = [
    "upsampling1d",
    "upsampling1d_add",
    "upsampling2d",
    "upsampling2d_add",
    "upsampling3d",
    "upsampling3d_add",
    "upsampling3d_compress_time",
]
