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
Pool module init.
"""

from honey.compiler.ops.pool.avg_pool1d import avg_pool1d
from honey.compiler.ops.pool.avg_pool2d import avg_pool2d
from honey.compiler.ops.pool.max_pool2d import max_pool2d


__all__ = ["avg_pool1d", "avg_pool2d", "max_pool2d"]
