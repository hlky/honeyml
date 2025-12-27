from dinoml.builder.base import Build, _model_name_with_resolution
from dinoml.mapping.esrgan import map_esrgan


class ESRGANBuilder(Build):
    """

    Example:
    ```

    builder = ESRGANBuilder(
        hf_hub="hlky/RealESRGAN_x4plus",
        label="x4plus",
        dtype="float16",
        device="cuda",
        build_kwargs={
            "batch_size": 1,
            "resolution": (8, 512),
        },
    )

    ```

    """

    model_name = "ESRGAN.{label}.{resolution}.{device_name}.sm{sm}"
    map_function = map_esrgan
    model_output_names = ["Y"]
    model_output = None

    _model_name = _model_name_with_resolution
