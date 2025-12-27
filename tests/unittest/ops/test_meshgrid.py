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


class MeshGridTestCase(unittest.TestCase):
    def _test_single_op(
        self,
        input_shapes,
        indexing="ij",
        test_name="meshgrid_fp16",
        dtype="float16",
    ):
        target = detect_target()
        inputs = [
            Tensor(
                shape=[IntVar(values=shape, name=f"input_{i}_dim0")],
                dtype=dtype,
                name=f"input_{i}",
                is_input=True,
            )
            for i, shape in enumerate(input_shapes)
        ]
        OP = ops.meshgrid(indexing=indexing)
        outputs = OP(*inputs)
        for i, output in enumerate(outputs):
            output._attrs["name"] = f"output_{i}"
            output._attrs["is_output"] = True
        module = compile_model(outputs, target, "./tmp", test_name)

        input_tensors = [
            get_random_torch_tensor([shape[0]], dtype=dtype) for shape in input_shapes
        ]
        output_tensors = torch.meshgrid(*input_tensors, indexing=indexing)
        input_data = [tensor.contiguous() for tensor in input_tensors]
        output_data = [
            torch.empty_like(tensor).contiguous() for tensor in output_tensors
        ]
        module.run_with_tensors(input_data, output_data)
        for out_dinoml, out_pt in zip(output_data, output_tensors):
            torch.testing.assert_close(
                out_dinoml,
                out_pt.to(out_dinoml.dtype),
                rtol=1e-3,
                atol=1e-3,
                msg=lambda msg: f"{msg}\n\n{test_name}\npt ({out_pt.shape}):\n{out_pt}\n\ndinoml ({out_dinoml.shape}):\n{out_dinoml}\n\n",
            )

    @parameterized.expand(
        **filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16")],
                # TestEnv.CUDA_LESS_THAN_SM80: [("float16"), ("float32")],
                # TestEnv.CUDA_SM80: [("bfloat16")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_meshgrid(self, dinoml_dtype):
        self._test_single_op(
            input_shapes=[[4], [5], [6]],
            indexing="ij",
            test_name=f"meshgrid_{dinoml_dtype}_ij",
            dtype=dinoml_dtype,
        )
        self._test_single_op(
            input_shapes=[[10], [20]],
            indexing="ij",
            test_name=f"meshgrid_{dinoml_dtype}_ij",
            dtype=dinoml_dtype,
        )
        self._test_single_op(
            input_shapes=[[10], [20]],
            indexing="xy",
            test_name=f"meshgrid_{dinoml_dtype}_xy",
            dtype=dinoml_dtype,
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()
