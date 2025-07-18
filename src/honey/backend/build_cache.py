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


from honey.backend.build_cache_base import (
    BuildCache,
    FileBasedBuildCache,
    NoBuildCache,
)
from honey.utils import environ as honey_env

__all__ = ["BUILD_CACHE", "BuildCache"]


def create_build_cache() -> BuildCache:
    build_cache_dir = honey_env.honey_build_cache_dir()
    if build_cache_dir is None or build_cache_dir == "":
        return NoBuildCache()
    else:
        return FileBasedBuildCache(build_cache_dir)


BUILD_CACHE: BuildCache = create_build_cache()
