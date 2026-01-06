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
from dinoml.compiler.base import Tensor
from dinoml.compiler.ops.gemm_universal.gemm_rcr_bias import gemm_rcr_bias
from dinoml.compiler.tensor_accessor import TensorAccessor


class dual_gemm_rcr_bias_fast_gelu(gemm_rcr_bias):
    def __init__(self):
        super().__init__()
        self._attrs["op"] = "dual_gemm_rcr_bias_fast_gelu"
        self._attrs["epilogue2"] = "LeftFastGeluAndMul"

    def _infer_shapes(self, a: Tensor, w1: Tensor, w2: Tensor, bias1: Tensor, bias2: Tensor):
        return super()._infer_shapes(a, w1, bias1)

    def __call__(
        self, a: Tensor, w1: Tensor, w2: Tensor, bias1: Tensor, bias2: Tensor
    ) -> Tensor:
        a, b = self._align_ab(a, w1)
        self._attrs["inputs"] = [a, w1, w2, bias1, bias2]
        self._attrs["input_accessors"] = [
            TensorAccessor(tensor) for tensor in self._attrs["inputs"]
        ]
        self._set_depth()
        self._sanity_check(a, w1)
        output_shape = self._infer_shapes(a, w1, w2, bias1, bias2)
        self._extract_epilogue_alignment(output_shape)
        output = Tensor(
            output_shape,
            src_ops={self},
            dtype=self._attrs["inputs"][0]._attrs["dtype"],
        )
        self._attrs["outputs"] = [output]
        self._attrs["output_accessors"] = [TensorAccessor(output)]
        if w1._attrs["shape"][-2] != 1 and w2._attrs["shape"][-2] == 1:
            self._attrs["broadcast_b1"] = True
        return output
