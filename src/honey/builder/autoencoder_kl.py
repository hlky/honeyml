from honey.builder.base import Build, _model_name_with_resolution
from honey.mapping.autoencoder_kl import map_autoencoder_kl


class AutoencoderKLDecodeBuilder(Build):
    """

    Example:
    ```

    builder = AutoencoderKLDecodeBuilder(
        hf_hub="runwayml/stable-diffusion-v1-5",
        label="v1",
        dtype="float16",
        device="cuda",
        build_kwargs={
            "batch_size": 1,
            "resolution": (8, 512),
        },
        model_kwargs={
            "subfolder": "vae",
        }
    )

    ```

    """

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
