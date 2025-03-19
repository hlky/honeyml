import unittest

from typing import cast, List, Optional, Tuple

import torch
from honey.compiler import compile_model, ops
from honey.frontend import nn, Tensor
from honey.testing import detect_target
from honey.testing.test_utils import get_random_torch_tensor, string_to_torch_dtype

from honey.utils.import_path import import_parent

if __name__ == "__main__":
    import_parent(filepath=__file__, level=1)

from utils import mark_output


class Float8TestModel(nn.Module):
    def __init__(self, dtype: str, shape: List[int]):
        super().__init__()
        self.weight = nn.Parameter(shape, dtype=dtype)

    def forward(self, x: Tensor):
        if self.weight.tensor().dtype() != x.dtype():
            return x + ops.cast()(self.weight.tensor(), x.dtype())
        return x + self.weight.tensor()


class Float8TestCase(unittest.TestCase):
    def _test_float8(
        self,
        shape: List[int],
        weight_dtype: str = "float8_e4m3",
        infer_dtype: Optional[str] = None,
        tolerance: float = 1e-5,
    ):
        if infer_dtype is None:
            infer_dtype = weight_dtype
        x = get_random_torch_tensor(shape).to(dtype=string_to_torch_dtype(infer_dtype))
        y = torch.empty_like(x)
        constants = {
            "weight": torch.randn(shape, device="cuda").to(
                dtype=string_to_torch_dtype(weight_dtype)
            )
        }
        print(constants)

        X = Tensor(shape=shape, dtype=infer_dtype, name="X", is_input=True)
        module = Float8TestModel(dtype=weight_dtype, shape=shape)
        module.name_parameter_tensor()
        Y = module.forward(X)
        Y = mark_output(Y, "Y")
        target = detect_target()
        module = compile_model(
            Y,
            target,
            "./tmp",
            f"test_{weight_dtype}-weight_{infer_dtype}-infer",
            constants=constants,
            do_constant_folding=False,
        )

        module.run_with_tensors([x], [y])
        print("x: ", x)
        print("y: ", y)
        y = y.float()
        y_pt = x.float() + constants["weight"].float()
        torch.testing.assert_close(
            y_pt,
            y,
            rtol=tolerance,
            atol=tolerance,
            msg=lambda msg: f"{msg}\n\npt ({y_pt.shape}):\n{y_pt}\n\nhoney ({y.shape}):\n{y}\n\n",
        )

    def test_float8(self):
        self._test_float8(
            shape=[64], tolerance=1e-3, weight_dtype="float16", infer_dtype=None
        )
        self._test_float8(
            shape=[64], tolerance=125e-3, weight_dtype="float8_e4m3", infer_dtype=None
        )
        self._test_float8(
            shape=[64],
            tolerance=1e-3,
            weight_dtype="float8_e4m3",
            infer_dtype="float16",
        )


if __name__ == "__main__":
    unittest.main()
