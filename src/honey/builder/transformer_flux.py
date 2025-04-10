from honey.builder.base import Build, _model_name_with_resolution
from honey.mapping.transformer_flux import map_transformer_flux


class FluxTransformer2DBuilder(Build):
    """

    Example:
    ```
        builder = FluxTransformer2DBuilder(
            hf_hub="black-forest-labs/FLUX.1-schnell",
            label="schnell",
            dtype="bfloat16",
            device="cuda",
            build_kwargs={
                "batch_size": (1, 2),
                "resolution": (512, 1024),
                "seq_len": 512,
                "ids_size": 3,
            },
            model_kwargs={
                "subfolder": "transformer",
            }
        )
    ```

    """

    model_name = "transformer_flux.{label}.{resolution}.{device_name}.sm{sm}"
    map_function = map_transformer_flux
    model_output_names = ["Y"]

    _model_name = _model_name_with_resolution
