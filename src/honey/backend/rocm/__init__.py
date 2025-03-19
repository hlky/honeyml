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
"""
Rocm backend init.
"""
from honey.backend.rocm import lib_template, target_def, utils
from honey.backend.rocm.attention import *
from honey.backend.rocm.common import *
from honey.backend.rocm.conv2d import *
from honey.backend.rocm.embedding import *
from honey.backend.rocm.gemm import *
from honey.backend.rocm.pool2d import *
from honey.backend.rocm.view_ops import *
from honey.backend.rocm.elementwise import *
from honey.backend.rocm.tensor import *
from honey.backend.rocm.normalization import softmax
from honey.backend.rocm.upsample import *
from honey.backend.rocm.vision_ops import *
from honey.backend.rocm.padding import *
from honey.backend.rocm.normalization import groupnorm, groupnorm_swish, layernorm
