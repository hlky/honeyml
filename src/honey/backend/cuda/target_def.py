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
CUDA target specialization
"""

import logging
import os
import re
import shutil
import sys

from typing import List

from honey.backend import registry


from honey.backend.target import (
    Honey_STATIC_FILES_PATH,
    CUTLASS_PATH,
    Target,
)

from honey.utils import environ
from honey.utils.misc import is_debug

# pylint: disable=C0415,W0707,W0611,W0702,W1401


_LOGGER = logging.getLogger(__name__)


class CUDA(Target):
    """CUDA target."""

    def __init__(
        self,
        template_path=CUTLASS_PATH,
        honey_static_files_path=Honey_STATIC_FILES_PATH,
        arch="80",
        cuda_version=None,
        **kwargs,
    ):
        """CUDA target init.

        Parameters
        ----------
        template_path : str, optional
            by default "${repo_root}/3rdparty/cutlass"
        honey_static_files_path : str
            Absolute path to the Honey static/ directory
        """
        super().__init__(honey_static_files_path)
        self._target_type = 1
        self._template_path = template_path
        self._honey_include_path = honey_static_files_path
        self._arch = arch
        self._kwargs = kwargs
        self._compile_options = self._build_compile_options()
        if cuda_version is None:
            # try to set default CUDA version based on the arch
            if arch == "80":
                cuda_version = "11.4.2"
            elif arch == "90":
                cuda_version = "12.0.0"
        self._cuda_version = cuda_version

    def _build_include_directories(self) -> List[str]:
        cutlass_path = [
            os.path.join(self._template_path, "include"),
            os.path.join(self._template_path, "tools/util/include"),
            os.path.join(self._template_path, "examples/35_gemm_softmax"),
            os.path.join(self._template_path, "examples/41_fused_multi_head_attention"),
            os.path.join(self._template_path, "examples/45_dual_gemm"),
        ]
        honey_kernels_static_path = os.path.join(
            self._honey_include_path, "include/kernels"
        )
        honey_static_path = os.path.join(self._honey_include_path, "include")

        output = [honey_kernels_static_path, honey_static_path]
        output.extend(cutlass_path)
        return output

    def get_include_directories(self) -> List[str]:
        return self._build_include_directories()

    def _build_gnu_host_compiler_options(self) -> List[str]:
        return [
            "-fPIC",
            "-Wconversion",
            "-fno-strict-aliasing",
            "-fvisibility=hidden",
        ]

    def get_host_compiler_options(self) -> List[str]:
        return self._build_gnu_host_compiler_options()

    def _get_nvcc_debug_options(self) -> List[str]:
        CUDA_DEBUG_LEVEL_STRINGS = [[], ["-lineinfo"], ["-g", "-G"]]
        level = environ.get_cuda_nvcc_debug_level()
        if level.isdigit():
            level = int(level)
            assert level >= 0 and level < 3, (
                "Debug level out of range. Must be 0 (no debug info), 1 (lineinfo) or 2 (with debug info, disable opt)"
            )
            return CUDA_DEBUG_LEVEL_STRINGS[level]
        return [level]

    def _build_nvcc_compiler_options(self) -> List[str]:
        code = [f"sm_{self._arch}", f"compute_{self._arch}"]
        if environ.enable_cuda_lto():
            code += [f"lto_{self._arch}"]
        options = [
            "-DHoney_CUDA",
            "-t=0",
            "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
            "-w",
            f"-gencode=arch=compute_{self._arch},code=[{','.join(code)}]",
            environ.get_compiler_opt_level(),
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "-DCUTLASS_DEBUG_TRACE_LEVEL=" + environ.get_cutlass_debug_trace_level(),
        ]
        if self._kwargs.get("optimize_for_compilation_time", True):
            options.append("-DOPTIMIZE_FOR_COMPILATION_TIME")
        if environ.enable_ptxas_info():
            (
                options.extend(
                    [
                        "--keep",  # Keep the intermediate files for debugging (including ptx, sass, cubin etc.)
                        "--ptxas-options=--warn-on-local-memory-usage",  # warn us if local memory is used in CUDA Kernels
                        "--ptxas-options=--warn-on-spills",  # warn us if register spilling happens in CUDA Kernels
                        "--resource-usage",  # Report on CUDA resource usage (shared mem, registers etc.)
                        "--source-in-ptx",
                    ]
                ),
            )  # Annotate the ptx file with source information
        options.extend(self._get_nvcc_debug_options())
        if self._ndebug == 1:
            options.append("-DNDEBUG")
        if environ.use_fast_math() and (
            "use_fast_math" not in self._kwargs or self._kwargs["use_fast_math"]
        ):
            options.extend(
                [
                    "--use_fast_math",
                    "-DHoney_USE_FAST_MATH=1",
                ]
            )
        if (
            self._kwargs.get("use_tanh_for_sigmoid", False)
            or environ.use_tanh_for_sigmoid()
        ):
            options.extend(
                [
                    "-DHoney_USE_TANH_FOR_SIGMOID=1",
                    "-DCUTLASS_USE_TANH_FOR_SIGMOID=1",
                ]
            )
        return options

    def get_device_compiler_options(self) -> List[str]:
        return self._build_nvcc_compiler_options()

    def _build_compile_options(self):
        include_paths = self._build_include_directories()
        host_compiler_options = self._build_gnu_host_compiler_options()
        nvcc_compiler_options = self._build_nvcc_compiler_options()

        options = (
            nvcc_compiler_options
            + [
                f"-Xcompiler {opt}" if "=" in opt else f"-Xcompiler={opt}"
                for opt in host_compiler_options
            ]
            + ["-I" + path for path in include_paths]
        )

        return " ".join(options)

    def src_extension(self):
        return ".cu"

    def _gen_cutlass_lib_pkg(self):
        self.lib_folder = "src/honey/utils/cutlass_lib"

    def __enter__(self):
        super().__enter__()
        self._gen_cutlass_lib_pkg()
        f_gen_ops = registry.get("cuda.gen_cutlass_ops")
        allow_cutlass_sm90 = (
            self._kwargs.get("allow_cutlass_sm90", False)
            or environ.allow_cutlass_sm90_kernels()
        )
        force_cutlass_sm90 = (
            self._kwargs.get("force_cutlass_sm90", False)
            or environ.force_cutlass_sm90_kernels()
        )
        self._operators = f_gen_ops(
            self._arch, self._cuda_version, allow_cutlass_sm90, force_cutlass_sm90
        )

    def cc(self):
        cc = "nvcc"
        if environ.nvcc_ccbin():
            cc += " -ccbin " + environ.nvcc_ccbin()
        return cc

    def compile_cmd(self, executable=False):
        if executable:
            cmd = self.cc() + " " + self._compile_options + " -o {target} {src}"
        else:
            cmd = self.cc() + " " + self._compile_options + " -c -o {target} {src}"
        return cmd

    def dev_select_flag(self):
        return "CUDA_VISIBLE_DEVICES"

    def select_minimal_algo(self, algo_names: List[str]):
        def comp_func(name):
            compute_args = re.findall(r"(\d+)x(\d+)_(\d+)x(\d+)", name)
            if len(compute_args) != 1:
                raise RuntimeError("Invalid cutlass op name")
            args = [int(x) for x in compute_args[0]]
            align_args = name.split("_")
            args.append(int(align_args[-2]))
            args.append(int(align_args[-1]))
            return tuple(args)

        return min(algo_names, key=comp_func)


@registry.reg("cuda.create_target")
def create_target(template_path, arch, **kwargs):
    return CUDA(template_path=template_path, arch=arch, **kwargs)
