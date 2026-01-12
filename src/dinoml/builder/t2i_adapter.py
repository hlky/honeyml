from dinoml.builder.base import Build, _model_name_with_resolution
from dinoml.mapping.t2i_adapter import map_t2i_adapter


class T2IAdapterBuilder(Build):
    """

    Example:
    ```
        builder = T2IAdapterBuilder(
            hf_hub="hlky/t2iadapter_canny_sd15v2",
            label="canny_sd15v2",
            dtype="float16",
            device="cuda",
            build_kwargs={
                "batch_size": (1, 2),
                "resolution": (8, 512),
            }
        )
    ```

    """

    model_name = "t2i_adapter.{label}.{resolution}.{device_name}.{sm}"
    map_function = map_t2i_adapter
    model_output_names = "down_intrablock_additional_residuals_{idx}"
    model_output = None

    _model_name = _model_name_with_resolution
