import unittest

from typing import cast, List, Optional, Tuple

import torch
from honey.compiler import compile_model, ops
from honey.frontend import Tensor
from honey.testing import detect_target
from honey.testing.test_utils import get_random_torch_tensor
from honey.modeling.diffusers.activations import (
    ApproximateGELU,
    GEGLU,
    GELU,
    get_activation,
)
from honey.builder.config import mark_output

from diffusers.models.activations import (
    ApproximateGELU as ApproximateGELU_torch,
    GEGLU as GEGLU_torch,
    GELU as GELU_torch,
    get_activation as get_activation_torch,
)


class ActivationsTestCase(unittest.TestCase):
    def _test_activation(
        self,
        activation_name: str,
        shape: List[int],
        dtype: str = "float16",
        tolerance: float = 1e-5,
    ):
        X = Tensor(shape=shape, dtype=dtype, name="X", is_input=True)
        activation = get_activation(activation_name)
        Y = activation(X)
        Y = mark_output(Y, "Y")
        target = detect_target()
        module = compile_model(Y, target, "./tmp", f"test_{activation_name}")

        x = get_random_torch_tensor(shape, dtype=dtype)
        activation_torch = get_activation_torch(activation_name)
        y_pt = activation_torch(x)
        y = torch.empty_like(y_pt)

        module.run_with_tensors([x], [y])
        torch.testing.assert_close(
            y,
            y_pt,
            rtol=tolerance,
            atol=tolerance,
            msg=lambda msg: f"{msg}\n\npt ({y_pt.shape}):\n{y_pt}\n\nhoney ({y.shape}):\n{y}\n\n",
        )

    def _test_activation_module(
        self,
        module_class,
        shape: List[int],
        dim_in: int,
        dim_out: int,
        approximate: str = "none",
        dtype: str = "float16",
        tolerance: float = 1e-5,
    ):
        x = get_random_torch_tensor(shape, dtype=dtype)
        if module_class == GELU:
            op = GELU_torch(dim_in, dim_out, approximate=approximate)
        elif module_class == GEGLU:
            op = GEGLU_torch(dim_in, dim_out)
        else:
            op = ApproximateGELU_torch(dim_in, dim_out)
        op = op.eval().to(x.device, x.dtype)

        with torch.inference_mode():
            y_pt = op.forward(x)
        y = torch.empty_like(y_pt)

        state_dict_pt = cast(dict[str, torch.Tensor], op.state_dict())
        state_dict_honey = {}
        for key, value in state_dict_pt.items():
            key_honey = key.replace(".", "_")
            if value.ndim == 4 and "weight" in key:
                value = value.permute(0, 2, 3, 1).contiguous()
            value = value.to(x.device, x.dtype)
            state_dict_honey[key_honey] = value

        X = Tensor(shape=shape, dtype=dtype, name="X", is_input=True)

        if module_class == GELU:
            op_honey = module_class(dim_in, dim_out, approximate=approximate)
        else:
            op_honey = module_class(dim_in, dim_out)
        op_honey.name_parameter_tensor()

        Y = op_honey.forward(X)
        Y = mark_output(Y, "Y")
        target = detect_target()
        module = compile_model(
            Y,
            target,
            "./tmp",
            f"test_{module_class.__name__}",
            constants=state_dict_honey,
        )

        module.run_with_tensors([x], [y])
        torch.testing.assert_close(
            y,
            y_pt,
            rtol=tolerance,
            atol=tolerance,
            msg=lambda msg: f"{msg}\n\npt ({y_pt.shape}):\n{y_pt}\n\nhoney ({y.shape}):\n{y}\n\n",
        )

    def test_swish(self):
        self._test_activation(
            activation_name="swish", shape=[2, 77, 768], dtype="float16", tolerance=1e-3
        )

    def test_silu(self):
        self._test_activation(
            activation_name="silu", shape=[2, 77, 768], dtype="float16", tolerance=1e-3
        )

    def test_mish(self):
        self._test_activation(
            activation_name="mish", shape=[2, 77, 768], dtype="float16", tolerance=2e-3
        )

    def test_gelu(self):
        self._test_activation(
            activation_name="gelu", shape=[2, 77, 768], dtype="float16", tolerance=1e-3
        )

    def test_relu(self):
        self._test_activation(
            activation_name="relu", shape=[2, 77, 768], dtype="float16", tolerance=1e-4
        )

    def test_gelu_module(self):
        for approximate in ["none", "tanh"]:
            self._test_activation_module(
                module_class=GELU,
                shape=[2, 77, 768],
                dim_in=768,
                dim_out=768,
                approximate=approximate,
                dtype="float16",
                tolerance=1e-3,
            )

    def test_geglu_module(self):
        self._test_activation_module(
            module_class=GEGLU,
            shape=[2, 77, 768],
            dim_in=768,
            dim_out=768,
            dtype="float16",
            tolerance=1e-3,
        )

    def test_approximate_gelu_module(self):
        self._test_activation_module(
            module_class=ApproximateGELU,
            shape=[2, 77, 768],
            dim_in=768,
            dim_out=768,
            dtype="float16",
            tolerance=1e-3,
        )


if __name__ == "__main__":
    unittest.main()
