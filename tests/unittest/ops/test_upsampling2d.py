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
import unittest

import torch

from honey.compiler import compile_model
from honey.frontend import IntVar, nn, Tensor
from honey.testing import detect_target
from honey.testing.test_utils import (
    filter_test_cases_by_params,
    get_random_torch_tensor,
    TestEnv,
)
from parameterized import parameterized


_DEFAULT_BATCH_SIZE = [1, 3]


class UpsamplingTestCase(unittest.TestCase):
    def _test_single_op(
        self,
        scale_factor=2.0,
        mode="bilinear",
        align_corners=False,
        batch_size=_DEFAULT_BATCH_SIZE,
        test_name="bilinear_upsampling2d_fp16",
        dtype="float16",
    ):
        channels = 1024
        HH, WW = 8, 8
        target = detect_target()
        X = Tensor(
            shape=[IntVar(values=batch_size, name="input_batch"), HH, WW, channels],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        OP = nn.Upsampling2d(
            scale_factor=scale_factor, mode=mode, align_corners=align_corners
        )
        Y = OP(X)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", test_name)

        for b in batch_size:
            X_pt = get_random_torch_tensor([b, channels, HH, WW], dtype=dtype)
            Y_pt = torch.nn.functional.interpolate(
                X_pt, scale_factor=scale_factor, mode=mode, align_corners=align_corners
            )
            x = torch.permute(X_pt, (0, 2, 3, 1)).contiguous()
            y = torch.empty_like(Y_pt).permute((0, 2, 3, 1)).contiguous()
            module.run_with_tensors([x], [y])
            y_transpose = torch.permute(y, (0, 3, 1, 2))
            torch.testing.assert_close(
                y_transpose,
                Y_pt.to(y.dtype),
                rtol=1e-3,
                atol=1e-3,
                msg=lambda msg: f"{msg}\n\n{test_name}\npt ({Y_pt.shape}):\n{Y_pt}\n\nhoney ({y_transpose.shape}):\n{y_transpose}\n\n",
            )

    @parameterized.expand(
        **filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16"), ("float32")],
                TestEnv.CUDA_SM80: [("bfloat16")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_upsampling2d_constructor(self, honey_dtype):
        # Currently upsampling2d bilinear does not support bfloat16.
        if honey_dtype != "bfloat16":
            self._test_single_op(
                scale_factor=3.5,
                mode="bilinear",
                test_name=f"bilinear_upsampling2d_{honey_dtype}",
                dtype=honey_dtype,
            )
            self._test_single_op(
                scale_factor=2.0,
                mode="bilinear",
                align_corners=True,
                test_name=f"bilinear_align_corners_upsampling2d_{honey_dtype}",
                dtype=honey_dtype,
            )
        self._test_single_op(
            scale_factor=2.0,
            mode="nearest-exact",
            align_corners=None,
            test_name=f"nearest-exact_upsampling2d_{honey_dtype}",
            dtype=honey_dtype,
        )
        self._test_single_op(
            scale_factor=2.0,
            mode="nearest",
            align_corners=None,
            test_name=f"nearest_upsampling2d_{honey_dtype}",
            dtype=honey_dtype,
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
