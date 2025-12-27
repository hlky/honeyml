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
Util functions for ROCM.
"""

import os
import pathlib
import re
import shutil
import tempfile

from dinoml.backend import registry

# from . import extra_conv_emit, extra_cutlass_generator, extra_enum

# pylint: disable=C0103,C0415,W0707


class Args:
    def __init__(self, arch):
        self.operations = "all"
        self.build_dir = ""
        self.curr_build_dir = ""
        self.rocm_version = "5.0.2"
        self.generator_target = ""
        self.architectures = arch
        self.kernels = "all"
        self.ignore_kernels = ""
        self.kernel_filter_file = None
        self.selected_kernel_list = None
        self.interface_dir = None
        self.filter_by_cc = True


@registry.reg("rocm.gen_ck_ops")
def gen_ops(arch):
    import ck_lib

    args = Args(arch)
    manifest = ck_lib.manifest.Manifest(args)
    try:
        func = getattr(ck_lib.generator, "Generate" + arch.upper())
        func(manifest, args.rocm_version)
    except AttributeError as exc:
        raise NotImplementedError(
            "Arch " + arch + " is not supported by current cklib lib."
        ) from exc
    return manifest.operations
