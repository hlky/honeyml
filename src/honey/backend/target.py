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
Target object for Honey.
"""
import logging
import os
import pathlib
import shutil
import tempfile
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

from honey.backend import registry
from honey.backend.profiler_cache import ProfileCacheDB
from honey.utils.misc import is_linux


_LOGGER = logging.getLogger(__name__)

CURRENT_PATH = pathlib.Path(__file__).parent.resolve()
ROOT_PATH = CURRENT_PATH.parent.parent.parent
_3RDPARTY_PATH = ROOT_PATH.joinpath("3rdparty/")
Honey_STATIC_FILES_PATH = str(ROOT_PATH.joinpath("csrc/"))
CUTLASS_PATH = str(_3RDPARTY_PATH.joinpath("cutlass/"))
COMPOSABLE_KERNEL_PATH = str(_3RDPARTY_PATH.joinpath("composable_kernel/"))
CUB_PATH = str(_3RDPARTY_PATH.joinpath("cub/"))

CURRENT_TARGET = None


class TargetType(IntEnum):
    """Enum for target type."""

    cuda = 1
    rocm = 2


class Target:
    def __init__(self, static_files_path: str):
        """
        Parameters
        ----------
        static_files_path : str
            Absolute path to the Honey static/ directory
        """
        self._target_type = -1
        self._template_path = ""
        self._compile_cmd = ""
        self._cache_path = ""
        self._profile_cache = None
        self.static_files_path = static_files_path

        ndebug_str = os.getenv("Honey_NDEBUG", "1")
        try:
            self._ndebug = int(ndebug_str)
        except ValueError:
            self._ndebug = 0

    def __enter__(self):
        """Enter the target context manager.

        This will set CURRENT_TARGET to this target.

        Raises
        ------
        RuntimeError
            If CURRENT_TARGET is already set, this will raise a RuntimeError.
        """
        self._load_profile_cache()
        global CURRENT_TARGET
        if CURRENT_TARGET is not None:
            raise RuntimeError("Target has been set.")
        assert self._target_type > 0
        CURRENT_TARGET = self

    def __exit__(self, ptype, value, trace):
        """Exit the target context manager."""
        self._profile_cache = None
        global CURRENT_TARGET
        CURRENT_TARGET = None

    @staticmethod
    def current():
        """Obtain the current target.

        Returns
        -------
        Target
            return the current target object.

        Raises
        ------
        RuntimeError
            If no target is set, this will raise a RuntimeError.
        """
        if CURRENT_TARGET is None:
            raise RuntimeError("Target is not set yet.")
        return CURRENT_TARGET

    def template_path(self) -> str:
        """Return CUTLASS/CK path for this target.

        Returns
        -------
        str
            Absolute path to the CUTLASS/CK template directory.
        """
        return self._template_path

    def get_custom_libs(self, absolute_dir, filename) -> str:
        filename = os.path.join(absolute_dir, filename)
        with open(filename) as f:
            res = f.read()
            return res

    def name(self) -> str:
        """Return the name of the target.

        Returns
        -------
        str
            The name of the target.
        """
        return TargetType(self._target_type).name

    def cc(self):
        """Compiler for this target.

        Raises
        ------
        NotImplementedError
            Need to be implemented by subclass.
        """
        raise NotImplementedError

    def make(self):
        make_path = shutil.which("make")
        return make_path if make_path is not None else "make"

    def cmake(self):
        cmake_path = shutil.which("cmake")
        return cmake_path if cmake_path is not None else "cmake"

    def compile_cmd(self, executable: bool = False):
        """Compile command string template for this target.

        Parameters
        ----------
        executable : bool, optional
            Whether the command with compile an executable object
            by default False

        Raises
        ------
        NotImplementedError
            Need to be implemented by subclass.
        """
        raise NotImplementedError

    def binary_compile_cmd(self):
        """
        A command that turns a raw binary file into an object file that
        can be linked into the executable.
        """
        cmd = "ld -r -b binary -o {target} {src}"
        # Support models with >2GB constants on Linux only
        if is_linux():
            cmd += (
                " && objcopy --rename-section"
                " .data=.lrodata,alloc,load,readonly,data,contents"
                " {target} {target}"
            )
        return cmd

    def compile_options(self) -> str:
        """Options for compiling the target.

        Returns
        -------
        str
        """
        return ""

    def src_extension(self):
        """Source file extension for this target.

        Returns
        -------
        str
            Source file extension for this target.
        """
        return NotImplementedError

    def dev_select_flag(self):
        """Environment variable to select the device.

        Returns
        -------
        str
            Environment variable to select the device.
        """
        return NotImplementedError

    def apply_op_rules(self, op_def):
        """Apply special rules to change template op definition
        Parameters
        ----------
        op_def : str
            Operator definition code string
        Returns
        -------
        str
            Modified op definition code string
        """
        return op_def

    def select_minimal_algo(self, algo_names: List[str]):
        """Select the minimal algorithm from the list of algorithms.

        This is used in CI to speed up the test without running actually profiling.

        Parameters
        ----------
        algo_names : List[str]
            All the available algorithm names for selection.
        """
        return NotImplementedError

    def trick_ci_env(self) -> bool:
        """Check if we want to trick in_ci_env to make it False.

        This is used in workers where we do not have control of CI_FLAG

        Returns
        -------
        bool
            Whether to trick ci env.
        """
        return os.environ.get("TRICK_CI_ENV", None) == "1"

    def in_ci_env(self) -> bool:
        """Check if the current environment is CI.

        Returns
        -------
        bool
            Returns True if env CI_FLAG=CIRCLECI and TRICK_CI_ENV is not set (or 0).
        """
        return os.environ.get("CI_FLAG", None) == "CIRCLECI" and not self.trick_ci_env()

    def disable_profiler_codegen(self) -> bool:
        """Whether to disable profiler codegen.

        disable profiler codegen completely in CI to speed up long running unittest

        Returns
        -------
        bool
            Whether to disable profiler codegen.
        """
        return (
            os.environ.get("DISABLE_PROFILER_CODEGEN", None) == "1"
            and not self.force_profile()
        )

    def force_profile(self) -> bool:
        """Whether to force profile.

        Force profiling regardless in_ci_env, disable_profiler_codegen

        Returns
        -------
        bool
            Whether to force profile.
        """
        return os.environ.get("FORCE_PROFILE", None) == "1"

    def use_dummy_profiling_results(self) -> bool:
        """Whether to use dummy profiling results."""
        # Whether to use dummy profiling results to speed up runs.
        return self.in_ci_env() and not self.force_profile()

    def _get_cache_file_name(self) -> str:
        """Get the cache file name for this target.

        Returns
        -------
        str
            The cache file name for this target.
        """
        # TODO: Add device name
        cache_file = "{dev_type}.db".format(dev_type=TargetType(self._target_type).name)
        return cache_file

    def _prepare_profile_cache_path(self) -> Optional[str]:
        """Prepare local profile cache for this target."""
        if self.use_dummy_profiling_results():
            _LOGGER.info("Escape loading profile cache when using dummy profiling")
            return None

        prefix = None
        if os.environ.get("CACHE_DIR", None):
            prefix = os.environ.get("CACHE_DIR", None)
        cache_file = self._get_cache_file_name()
        if prefix is None:
            prefix = os.path.join(pathlib.Path.home(), ".honey")

        try:
            os.makedirs(prefix, exist_ok=True)
        except OSError as error:
            _LOGGER.info(f"Cannot mkdir at {prefix} due to issue {error}")
            prefix = os.path.join(tempfile.mkdtemp(prefix="honey_"), ".honey")
            os.makedirs(prefix, exist_ok=True)
            _LOGGER.info(f"mkdir at {prefix} instead")

        cache_path = os.path.join(prefix, cache_file)
        flush_flag = os.environ.get("FLUSH_PROFILE_CACHE", "0")
        if flush_flag != "0":
            os.remove(cache_path)
        return cache_path

    def _load_profile_cache(self):
        """Load local profile cache for this target."""
        self._cache_path = self._prepare_profile_cache_path()
        if self._cache_path is None:
            return

        _LOGGER.info(f"Loading profile cache from: {self._cache_path}")
        self._profile_cache = ProfileCacheDB(
            TargetType(self._target_type).name, path=self._cache_path
        )

    def get_profile_cache_path(self):
        """Get local profile cache path for this target."""
        return self._cache_path

    def get_profile_cache_version(self, op_class: str) -> int:
        """Get the current profile cache version for the op_class.

        Parameters
        ----------
        op_class : str
            Op class name: only gemm is supported at the moment.

        Returns
        -------
        int
            cache version.

        Raises
        ------
        NotImplementedError
            If op class is not supported, raise error.
        """
        # TODO: support conv and normalization
        if op_class == "gemm":
            return self._profile_cache.gemm_cache_version
        elif op_class == "conv":
            return self._profile_cache.conv_cache_version
        elif op_class == "conv3d":
            return self._profile_cache.conv3d_cache_version
        raise NotImplementedError

    def query_profile_cache(
        self, op_class: str, args: Dict[str, Any]
    ) -> Tuple[str, int]:
        """Query the profile cache for the given op class and args.

        Parameters
        ----------
        op_class : str
            Op class name. gemm, conv or normalization
        args : Dict[str, Any]
            Op arguments.

        Returns
        -------
        Tuple[str, int]
            Queried best profiling results.

        Raises
        ------
        NotImplementedError
            If op class is not supported, raise error.
        """
        if op_class == "gemm":
            return self._profile_cache.query_gemm(args)
        if op_class == "conv":
            return self._profile_cache.query_conv(args)
        if op_class == "conv3d":
            return self._profile_cache.query_conv3d(args)
        if op_class == "normalization":
            return self._profile_cache.query_normalization(args)
        raise NotImplementedError

    def insert_profile_cache(self, op_class: str, args: Dict[str, Any]):
        """Insert the profile cache for the given op class and args."""
        if op_class == "gemm":
            self._profile_cache.insert_gemm(args)
        elif op_class == "conv":
            self._profile_cache.insert_conv(args)
        elif op_class == "conv3d":
            self._profile_cache.insert_conv3d(args)
        elif op_class == "normalization":
            self._profile_cache.insert_normalization(args)
        else:
            raise NotImplementedError

    def copy_headers_and_csrc_to_workdir(self, workdir: str) -> List[str]:
        """
        Copy over all the files in include/ and csrc/ to some working directory.
        Skips files that are not marked with .cpp/.h

        Returns a list of copied source files (to be built later).

        Parameters
        ----------
        workdir : str
            The path to copy to
        """
        sources = []
        csrc = self.static_files_path
        for fname in os.listdir(csrc):
            fname_dst, ext = os.path.splitext(fname)
            if ext != ".cpp":
                continue
            # TODO: Remove this file when the linker error gets fixed in rocm backend.
            # All files in csrc should be shared between the ROCM and CUDA backends.
            if fname == "rocm_hack.cpp" and self.name() != "rocm":
                continue
            fname_src = os.path.join(csrc, fname)
            fname_dst_cpp = os.path.join(workdir, f"{fname_dst}{self.src_extension()}")
            shutil.copyfile(fname_src, fname_dst_cpp)
            sources.append(fname_dst_cpp)

        headers = []
        include = os.path.join(self.static_files_path, "include")
        for fname in os.listdir(include):
            _, ext = os.path.splitext(fname)
            if ext != ".h":
                continue
            fname_src = os.path.join(include, fname)
            fname_dst = os.path.join(workdir, fname)
            shutil.copyfile(fname_src, fname_dst)
            headers.append(fname_dst)
        return sources

    @classmethod
    def remote_logger(cls, record: Dict[str, Any]) -> None:
        """
        Upload the record remotely to some logging table.

        Parameters
        ----------
        record : Dict[str, Any]
            The dictionary storing the record
        """
        return

    def get_include_directories(self) -> List[str]:
        """
        Returns a list of include directories for a compiler.

        Raises
        ------
        NotImplementedError
            Need to be implemented by subclass.
        """
        raise NotImplementedError

    def get_host_compiler_options(self) -> List[str]:
        """
        Returns a list of options for the host compiler.

        Raises
        ------
        NotImplementedError
            Need to be implemented by subclass.
        """
        raise NotImplementedError

    def get_device_compiler_options(self) -> List[str]:
        """
        Returns a list of options for the device compiler.

        Raises
        ------
        NotImplementedError
            Need to be implemented by subclass.
        """
        raise NotImplementedError

    def postprocess_build_dir(self, build_dir: str) -> None:
        """
        Postprocess a build directory, allows final modification of the build directory before building.

        """
        pass


def CUDA(template_path: str = CUTLASS_PATH, arch: str = "80", **kwargs):
    """Create a CUDA target."""
    func = registry.get("cuda.create_target")
    return func(template_path, arch, **kwargs)


def ROCM(template_path: str = COMPOSABLE_KERNEL_PATH, arch: str = "gfx908", **kwargs):
    """Create a ROCM target."""
    func = registry.get("rocm.create_target")
    return func(template_path, arch, **kwargs)
