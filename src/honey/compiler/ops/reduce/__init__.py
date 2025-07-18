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
Reduce module init.
"""
from honey.compiler.ops.reduce.reduce_max import reduce_max
from honey.compiler.ops.reduce.reduce_mean import reduce_mean
from honey.compiler.ops.reduce.reduce_min import reduce_min
from honey.compiler.ops.reduce.reduce_sum import reduce_sum
from honey.compiler.ops.reduce.var import var
from honey.compiler.ops.reduce.vector_norm import vector_norm


__all__ = [
    "reduce_max",
    "reduce_mean",
    "reduce_min",
    "reduce_sum",
    "var",
    "vector_norm",
]
