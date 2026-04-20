# Modeling

In DinoML, "modeling" means describing a model with `dinoml.frontend.nn` modules and `dinoml.compiler.ops`, then compiling the symbolic graph into a standalone shared library.

The main flow is:

1. Define a DinoML module whose `forward()` signature uses `typing.Annotated` and `Shape(...)` metadata.
2. Provide symbolic build values such as `batch_size`, `resolution`, or `seq_len`.
3. Create symbolic input `Tensor`s from those annotations.
4. Run the model once with symbolic tensors to build the graph.
5. Mark outputs and call `compile_model(...)`.

## Tensor Layout

Most vision models in DinoML use channel-last tensors:

- 2D tensors: `(batch, height, width, channels)` / NHWC
- 3D tensors: `(batch, depth, height, width, channels)` / NDHWC

This differs from PyTorch's default NCHW layout. Mapping functions in `src/dinoml/mapping/` usually permute pretrained weights into DinoML layout before compilation.

## Shape Annotations

Builders derive inputs from `forward()` annotations with `build_tensors_from_annotations(...)` in `src/dinoml/utils/build_utils.py`.

```python
from typing import Annotated

from dinoml.frontend import Tensor
from dinoml.utils.build_utils import DimDiv, Shape

def forward(
    sample: Annotated[
        Tensor,
        (
            Shape(name="batch_size"),
            Shape(name="height", dim_operations=(DimDiv(8),)),
            Shape(name="width", dim_operations=(DimDiv(8),)),
            Shape(name="channels", config_name="in_channels"),
        ),
    ],
):
    ...
```

Rules:

- `Shape(name="batch_size")` reads from `build_kwargs["batch_size"]`.
- `config_name="in_channels"` reads from the loaded model config instead of `build_kwargs`.
- `DimAdd`, `DimSub`, `DimMul`, and `DimDiv` transform symbolic values before tensor creation.
- Passing a tuple such as `(1, 4)` creates an `IntVar`, which becomes a dynamic dimension.
- Annotated `Dict[str, Tensor]` inputs are supported and are used for nested inputs such as `added_cond_kwargs`.

## Builder Flow

Reusable builders live in `src/dinoml/builder/`. `Build.__call__()` performs the full compile pipeline:

1. `load_config(...)` downloads or loads `config.json` and resolves the DinoML class through `_CLASS_MAPPING`.
2. `create_module()` instantiates the DinoML module and names parameter tensors.
3. `create_input_tensors()` builds symbolic inputs from the annotated `forward()` signature.
4. `create_output_tensors()` runs the symbolic forward pass and marks graph outputs.
5. `create_constants()` loads pretrained PyTorch weights and converts them with a mapping function.
6. `compile()` calls `compile_model(...)`, writes generated sources under `./tmp/<model_name>/`, and can benchmark the compiled module.

Example:

```python
from dinoml.builder.autoencoder_kl import AutoencoderKLDecodeBuilder

builder = AutoencoderKLDecodeBuilder(
    hf_hub="runwayml/stable-diffusion-v1-5",
    label="v1",
    dtype="float16",
    device="cuda",
    build_kwargs={
        "batch_size": (1, 2),
        "resolution": (64, 512),
    },
    model_kwargs={"subfolder": "vae"},
)

module = builder()
```

`resolution` is expanded into `height` and `width` by the builder, while tuple values create dynamic ranges.

## Manual Graph Construction

Top-level scripts in `builder/` show the lower-level path. The manual workflow is:

1. Instantiate a DinoML module from `src/dinoml/modeling/`.
2. Create input `Tensor`s directly.
3. Call the module with symbolic tensors.
4. Mark the outputs with `mark_output(...)`.
5. Compile with `compile_model(...)`.

Use this path when experimenting with new ops, debugging graph generation, or porting a new architecture before creating a reusable builder.

## `compile_model(...)`

`compile_model(...)` in `src/dinoml/compiler/compiler.py` is the handoff from modeling to compilation. It performs graph toposort, constant binding, graph optimization, profiling, constant folding, memory planning, code generation, and shared-library build steps.

The main outputs are written to `./tmp/<test_name>/` and include graph debug dumps plus generated source/build artifacts. This directory is the first place to inspect when a builder compiles incorrectly or profiling fails.

For the full compile/runtime boundary, generated files, and C++ runtime responsibilities, see [ARCHITECTURE](./ARCHITECTURE.md).

## Where To Look Next

- `src/dinoml/modeling/diffusers/autoencoders/autoencoder_kl.py` for a representative annotated model
- `src/dinoml/modeling/diffusers/unets/unet_2d_condition.py` for nested and optional inputs
- `src/dinoml/builder/base.py` for the reusable builder pipeline
- `src/dinoml/builder/config.py` for Hugging Face config loading and class mapping
- `builder/*.py` for lower-level one-off compilation examples
