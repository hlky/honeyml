from dinoml.builder.base import Build, _model_name_with_resolution
from dinoml.mapping.unet2d_condition import map_unet2d_condition


class UNet2DConditionBuilder(Build):
    """

    Example:
    ```
        builder = UNet2DConditionBuilder(
            hf_hub="runwayml/stable-diffusion-v1-5",
            label="v1",
            dtype="float16",
            device="cuda",
            build_kwargs={
                "batch_size": (1, 2),
                "resolution": (8, 512),
                "seq_len": 77,
            },
            model_kwargs={
                "subfolder": "unet",
            }
        )
    ```

    """

    model_name = "unet2d_condition.{label}.{resolution}.{device_name}.sm{sm}"
    map_function = map_unet2d_condition
    model_output_names = ["Y"]

    _model_name = _model_name_with_resolution
