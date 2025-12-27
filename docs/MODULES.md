# Modules

## Builders

Builders are implemented in `dinoml/builder`.

### Example

```python

class AutoencoderKLDecodeBuilder(Build):
    model_name = "autoencoder_kl.{model_type}.{label}.{resolution}.{device_name}.sm{sm}"
    model_type = "decode"
    map_function = map_autoencoder_kl
    map_function_skip_keys = (
        "quant.",
        "encoder.",
    )
    model_forward = "_decode"
    model_output_names = ["Y"]

    _model_name = _model_name_with_resolution
```

Builders subclass `Build` which provides base functionality.

- `model_name`: This is how the compiled module is named. Format strings e.g. `{label}` and `{device_name}` are replaced internally.
- `model_type`: This is used in the `model_name`.
- `map_function`: This is a function used to map the weights and keys to DinoML format.
- `map_function_skip_keys`: This allows certain weights to be skipped by the mapping function.
- `model_forward`: Overrides the function to use in DinoML modeling if it's not `forward`
- `model_output_names`: Names the outputs, used when calling the module.

Example usage

```python
builder = AutoencoderKLDecodeBuilder(
    hf_hub="runwayml/stable-diffusion-v1-5",
    label="v1",
    dtype="float16",
    device="cuda",
    build_kwargs={
        "batch_size": 1,
        "resolution": (64, 512),
    },
    model_kwargs={
        "subfolder": "vae",
    }
)
```

- `hf_hub`: HuggingFace Hub repo for the model checkpoint - this provides model configuration and weights
- `label`: Label for the module, a friendly name that represents the model version etc.
- `dtype`: The datatype the module should be compiled for.
- `device`: The device used for profiling.
- `build_kwargs`: This sets the target shape for the module. See [MODELING](./MODELING.md)
- `model_kwargs`: These are used to get the checkpoint config and weights from the Hub.

## Scripts

Scripts in `scripts/` wrap the Builders for ease-of-use.

### Example

```sh
python scripts/autoencoder_kl_decode_build.py --hf-hub runwayml/stable-diffusion-v1-5 --subfolder vae --label v1 --batch-size 1 --min-res 64 --max-res 512
```
