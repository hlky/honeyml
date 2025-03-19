import torch

from honey.compiler import compile_model, ops
from honey.frontend import Tensor
from honey.testing import detect_target


def verify_repeat_interleave_rank2():
    shapes = [4, 3]
    repeats = 2
    repeat_dim = 1
    x = torch.randn(shapes).cuda().half()
    y_pt = x.repeat_interleave(repeats, dim=repeat_dim)
    y = torch.empty_like(y_pt)
    X = Tensor(
        shape=shapes,
        dtype="float16",
        name="X",
        is_input=True,
    )
    Y = ops.repeat_interleave(repeats, repeat_dim)(X)
    print(Y.shape())
    Y._attrs["is_output"] = True
    Y._attrs["name"] = "Y"
    target = detect_target()
    with compile_model(Y, target, "./tmp", "repeat_interleave") as module:
        inputs = {"X": x}
        outputs = {"Y": y}
        module.run_with_tensors(inputs, outputs)
        tolerance = 0.0
        torch.testing.assert_close(
            y,
            y_pt.to(y.dtype),
            rtol=tolerance,
            atol=tolerance,
            msg=lambda msg: f"{msg}\n\npt ({y_pt.shape}):\n{y_pt}\n\nhoney ({y.shape}):\n{y}\n\n",
        )


def verify_repeat_interleave_rank3():
    shapes = [2, 33, 768]
    repeats = 3
    repeat_dim = 1
    x = torch.randn(shapes).cuda().half()
    y_pt = x.repeat_interleave(repeats, dim=repeat_dim)
    y = torch.empty_like(y_pt)
    X = Tensor(
        shape=shapes,
        dtype="float16",
        name="X",
        is_input=True,
    )
    Y = ops.repeat_interleave(repeats, repeat_dim)(X)
    print(Y.shape())
    Y._attrs["is_output"] = True
    Y._attrs["name"] = "Y"
    target = detect_target()
    with compile_model(Y, target, "./tmp", "repeat_interleave") as module:
        inputs = {"X": x}
        outputs = {"Y": y}
        module.run_with_tensors(inputs, outputs)
        tolerance = 0.0
        torch.testing.assert_close(
            y,
            y_pt.to(y.dtype),
            rtol=tolerance,
            atol=tolerance,
            msg=lambda msg: f"{msg}\n\npt ({y_pt.shape}):\n{y_pt}\n\nhoney ({y.shape}):\n{y}\n\n",
        )


def verify_repeat_interleave_rank4():
    shapes = [1, 4, 64, 64]
    b, c, h, w = shapes
    repeats = 2
    repeat_dim = 2  # h
    repeat_dim_honey = 1
    x = torch.randn(shapes).cuda().half()
    x_honey = x.clone().permute(0, 2, 3, 1).contiguous()
    y_pt = x.repeat_interleave(repeats, dim=repeat_dim)
    y = torch.empty_like(y_pt.permute(0, 2, 3, 1).contiguous())
    X = Tensor(
        shape=[b, h, w, c],
        dtype="float16",
        name="X",
        is_input=True,
    )
    Y = ops.repeat_interleave(repeats, repeat_dim_honey)(X)
    print(Y.shape())
    Y._attrs["is_output"] = True
    Y._attrs["name"] = "Y"
    target = detect_target()
    with compile_model(Y, target, "./tmp", "repeat_interleave") as module:
        inputs = {"X": x_honey}
        outputs = {"Y": y}
        module.run_with_tensors(inputs, outputs)
        tolerance = 0.0
        y = y.permute(0, 3, 1, 2).contiguous()
        torch.testing.assert_close(
            y,
            y_pt.to(y.dtype),
            rtol=tolerance,
            atol=tolerance,
            msg=lambda msg: f"{msg}\n\npt ({y_pt.shape}):\n{y_pt}\n\nhoney ({y.shape}):\n{y}\n\n",
        )


# verify_repeat_interleave_rank2()
verify_repeat_interleave_rank3()
verify_repeat_interleave_rank4()
