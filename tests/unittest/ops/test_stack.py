import unittest

import torch

from dinoml.compiler import compile_model, ops
from dinoml.frontend import IntVar, Tensor
from dinoml.testing import detect_target
from dinoml.testing.test_utils import (
    filter_test_cases_by_params,
    get_random_torch_tensor,
    TestEnv,
)
from parameterized import parameterized


_DEFAULT_BATCH_SIZE = [1, 3]


class StackTestCase(unittest.TestCase):
    def _test_stack(
        self,
        shape,
        dim,
        count=2,
        batch_size=_DEFAULT_BATCH_SIZE,
        test_name="stack_fp16",
        dtype="float16",
    ):
        target = detect_target()
        inputs = [
            Tensor(
                shape=[IntVar(values=batch_size, name="input_batch"), *shape],
                dtype=dtype,
                name=f"input_{idx}",
                is_input=True,
            )
            for idx in range(count)
        ]
        OP = ops.stack()
        Y = OP(inputs, dim=dim)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", test_name)

        for b in batch_size:
            inputs_pt = [
                get_random_torch_tensor([b, *shape], dtype=dtype) for _ in range(count)
            ]
            Y_pt = torch.stack(inputs_pt, dim=dim)
            x = inputs_pt
            y = torch.empty_like(Y_pt)
            module.run_with_tensors(x, [y])
            torch.testing.assert_close(
                y,
                Y_pt.to(y.dtype),
                rtol=1e-3,
                atol=1e-3,
                msg=lambda msg: f"{msg}\n\n{test_name}\npt ({Y_pt.shape}):\n{Y_pt}\n\ndinoml ({y.shape}):\n{y}\n\n",
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
    def test_stack(self, dinoml_dtype):
        # shape excludes batch
        self._test_stack(
            shape=[],
            dim=1,
            test_name=f"stack_{dinoml_dtype}",
            dtype=dinoml_dtype,
        )
        self._test_stack(
            shape=[3],
            dim=1,
            test_name=f"stack_{dinoml_dtype}",
            dtype=dinoml_dtype,
        )
        self._test_stack(
            shape=[2, 3],
            dim=-1,
            test_name=f"stack_{dinoml_dtype}",
            dtype=dinoml_dtype,
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
