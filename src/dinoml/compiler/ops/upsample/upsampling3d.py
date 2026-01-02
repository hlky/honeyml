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

from dinoml.compiler.ops.upsample.upsampling3d_base import upsampling3d_base


class upsampling3d(upsampling3d_base):
    def __init__(self, scale_factor, mode, align_corners=False) -> None:
        super().__init__(scale_factor, mode, align_corners)
        self._attrs["op"] = "upsampling3d"
        self._attrs["mode"] = mode
