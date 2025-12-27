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
Automatic detect target for testing
"""

import logging
import os
from subprocess import PIPE, Popen

from dinoml.backend.target import CUDA, ROCM

# pylint: disable=W0702, W0612,R1732


_LOGGER = logging.getLogger(__name__)

IS_CUDA = None
FLAG = ""


def _detect_cuda_with_nvidia_smi():
    try:
        proc = Popen(
            ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv"],
            stdout=PIPE,
            stderr=PIPE,
        )
        stdout, stderr = proc.communicate()
        stdout = stdout.decode("utf-8")
        sm_names = {
            "70": ["V100"],
            "75": ["T4", "Quadro T2000"],
            "80": ["PG509", "A100", "A800", "A10G", "RTX 30", "A30", "RTX 40"],
            "90": ["H100", "H800"],
        }
        for sm, names in sm_names.items():
            if any(name in stdout for name in names):
                return sm
        return None
    except Exception:
        return None


def _detect_cuda():
    import torch

    major, minor = torch.cuda.get_device_capability()
    comp_cap = major * 10 + minor
    if comp_cap >= 90:
        return "90"
    elif comp_cap >= 80:
        return "80"
    elif comp_cap >= 75:
        return "75"
    elif comp_cap >= 70:
        return "70"
    else:
        _LOGGER.warning(
            f"Unsupported platform, got {comp_cap=}, expected >= 7.0. Overriding to `8.0` to allow debugging. Compilation will likely fail."
        )
        return "80"


def _detect_rocm():
    try:
        proc = Popen(["rocminfo"], stdout=PIPE, stderr=PIPE)
        stdout, stderr = proc.communicate()
        stdout = stdout.decode("utf-8")
        if "gfx90a" in stdout:
            return "gfx90a"
        if "gfx908" in stdout:
            return "gfx908"
        return None
    except Exception:
        return None


def detect_target(**kwargs):
    """Detect GPU target based on nvidia-smi and rocminfo

    Returns
    -------
    Target
        CUDA or ROCM target

    """
    global IS_CUDA, FLAG
    if FLAG:
        if IS_CUDA:
            return CUDA(arch=FLAG, **kwargs)
        else:
            return ROCM(arch=FLAG, **kwargs)
    doc_flag = os.getenv("DINOML_BUILD_DOCS", None)
    if doc_flag is not None:
        return CUDA(arch="80", **kwargs)
    flag = _detect_cuda()
    if flag is not None:
        IS_CUDA = True
        FLAG = flag

        _LOGGER.info("Set target to CUDA")
        return CUDA(arch=flag, **kwargs)
    flag = _detect_rocm()
    if flag is not None:
        IS_CUDA = False
        FLAG = flag

        _LOGGER.info("Set target to ROCM")
        return ROCM(arch=flag, **kwargs)
    raise RuntimeError("Unsupported platform")
