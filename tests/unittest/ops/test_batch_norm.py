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
from dinoml.compiler import compile_model

from dinoml.frontend import Tensor
from dinoml.frontend.nn import batch_norm
from dinoml.testing import detect_target
from dinoml.testing.test_utils import (
    get_random_torch_tensor,
    get_torch_empty_tensor,
)


class BatchnormTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)

    def __init__(self, *args, **kwargs):
        super(BatchnormTestCase, self).__init__(*args, **kwargs)
        self.test_id = 0

    def _test_batchnorm(
        self,
        num_features,
        bn_op,
        input_shape,
        input_type="float16",
        test_name="batch_norm",
    ):
        pt_op = getattr(torch.nn, bn_op)(num_features).cuda().half().eval()
        dinoml_op = getattr(batch_norm, bn_op)(
            num_features, eps=pt_op.eps, permute_input_output=True
        )
        dinoml_op.name_parameter_tensor()

        pt_params = dict(pt_op.named_parameters())
        pt_buffers = dict(pt_op.named_buffers())
        params_dinoml = {}
        for key, arr in pt_params.items():
            print(key, arr.shape)
            params_dinoml[key] = arr
        for key, arr in pt_buffers.items():
            print(key, arr.shape)
            params_dinoml[key] = arr

        X_pt = get_random_torch_tensor(input_shape, input_type)
        Y_pt = pt_op(X_pt)
        X_dinoml = Tensor(
            shape=input_shape, dtype=input_type, name="input0", is_input=True
        )
        Y_dinoml = dinoml_op(X_dinoml)

        Ys_dinoml = [var._attrs["values"][0] for var in Y_dinoml._attrs["shape"]]
        self.assertEqual(list(Y_pt.shape), Ys_dinoml)

        Y_dinoml._attrs["is_output"] = True
        Y_dinoml._attrs["name"] = "output"

        target = detect_target()
        module = compile_model(
            Y_dinoml,
            target,
            "./tmp",
            f"{test_name}_{self.test_id}",
            constants=params_dinoml,
        )
        self.test_id += 1

        y = get_torch_empty_tensor(Ys_dinoml, dtype=input_type)
        inputs = {"input0": X_pt}
        module.run_with_tensors(inputs, [y])

        print(f"PT output: {Y_pt=}")
        print(f"DinoML output: {y=}")
        self.assertTrue(torch.allclose(Y_pt, y, atol=1e-5, rtol=1e-5))

    def test_batch_norm(self):
        self._test_batchnorm(num_features=3, bn_op="BatchNorm1d", input_shape=[5, 3])
        self._test_batchnorm(
            num_features=3,
            bn_op="BatchNorm1d",
            input_shape=[5, 3, 234],
            test_name="batch_norm_1d",
        )
        self._test_batchnorm(
            num_features=3,
            bn_op="BatchNorm2d",
            input_shape=[1, 3, 244, 244],
            test_name="batch_norm_2d",
        )
        self._test_batchnorm(
            num_features=6,
            bn_op="BatchNorm3d",
            input_shape=[4, 6, 24, 24, 11],
            test_name="batch_norm_3d",
        )


if __name__ == "__main__":
    unittest.main()
