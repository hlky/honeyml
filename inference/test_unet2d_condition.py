"""
Smoke test for a compiled UNet2DCondition DinoML module.

By default this runs the max-shape 512 latent path without optional residual
inputs. Set INCLUDE_OPTIONAL_RESIDUALS = True to exercise the residual path.
"""

from pathlib import Path

import torch

from dinoml.compiler import Model
from dinoml.compiler.dtype import _ENUM_TO_TORCH_DTYPE


MODEL_NAME = "unet2d_condition.v1.512.rtx_a4500.sm86"
MODEL_PATH = Path("./tmp") / MODEL_NAME / f"{MODEL_NAME}.so"

DEVICE = "cuda"
BATCH_SIZE = 1
INCLUDE_OPTIONAL_RESIDUALS = False
USE_ZEROS = True
RUN_BENCHMARK = False
COUNT = 50
REPEAT = 3
GRAPH_MODE = False
SYNC = True


def _make_tensor(name: str, shape, dtype: torch.dtype) -> torch.Tensor:
    if name == "timestep":
        return torch.zeros(shape, dtype=dtype, device=DEVICE)
    if dtype.is_floating_point:
        factory = torch.zeros if USE_ZEROS else torch.randn
        return factory(shape, dtype=dtype, device=DEVICE)
    return torch.randint(0, 64, shape, dtype=dtype, device=DEVICE)


def _make_inputs(module: Model):
    name_to_index = module.get_input_name_to_index_map()
    optional_map = module._input_name_to_optional

    inputs = {}
    for name, idx in sorted(name_to_index.items(), key=lambda item: item[1]):
        if optional_map[name] and not INCLUDE_OPTIONAL_RESIDUALS:
            inputs[name] = None
            continue

        shape = module.get_input_maximum_shape(idx)
        shape[0] = BATCH_SIZE
        dtype = _ENUM_TO_TORCH_DTYPE[module.get_input_dtype(idx)]
        inputs[name] = _make_tensor(name, shape, dtype)
    return inputs


def _make_outputs(module: Model):
    outputs = {}
    for name, idx in sorted(
        module.get_output_name_to_index_map().items(), key=lambda item: item[1]
    ):
        shape = module.get_output_maximum_shape(idx)
        shape[0] = BATCH_SIZE
        dtype = _ENUM_TO_TORCH_DTYPE[module.get_output_dtype(idx)]
        outputs[name] = torch.empty(shape, dtype=dtype, device=DEVICE)
    return outputs


def main():
    module = Model(str(MODEL_PATH))
    inputs = _make_inputs(module)
    outputs = _make_outputs(module)

    required_inputs = [
        name for name, is_optional in module._input_name_to_optional.items() if not is_optional
    ]
    optional_inputs = [
        name for name, is_optional in module._input_name_to_optional.items() if is_optional
    ]

    print(f"Loaded: {MODEL_PATH}")
    print(f"Required inputs: {required_inputs}")
    print(
        f"Optional residuals: {'enabled' if INCLUDE_OPTIONAL_RESIDUALS else 'disabled'} ({len(optional_inputs)} tensors)"
    )

    if RUN_BENCHMARK:
        mean, std, _ = module.benchmark_with_tensors(
            inputs=inputs,
            outputs=outputs,
            count=COUNT,
            repeat=REPEAT,
            graph_mode=GRAPH_MODE,
        )
        print(f"Mean: {mean:.3f} ms, Std: {std:.3f} ms")
        return

    outputs = module.run_with_tensors(
        inputs=inputs,
        outputs=outputs,
        sync=SYNC,
        graph_mode=GRAPH_MODE,
    )
    y = outputs["Y"]
    print(f"Y shape: {list(y.shape)}")
    print(f"Y dtype: {y.dtype}")
    print(f"Y abs max: {y.abs().max().item():.6f}")
    print(f"Y mean: {y.float().mean().item():.6f}")


if __name__ == "__main__":
    main()
