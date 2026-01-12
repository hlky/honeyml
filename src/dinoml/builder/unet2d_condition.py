from dinoml.builder.base import Build, _model_name_with_resolution
from dinoml.frontend import Tensor
from dinoml.mapping.unet2d_condition import map_unet2d_condition
from dinoml.utils.build_utils import build_tensors_from_annotations
from dinoml.utils.shape_utils import get_shape


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

    model_name = "unet2d_condition.{label}.{resolution}.{device_name}.{sm}"
    map_function = map_unet2d_condition
    model_output_names = ["Y"]

    _model_name = _model_name_with_resolution

    def create_input_tensors(self):
        self.input_tensors = build_tensors_from_annotations(
            getattr(self.dinoml_module, self.model_forward),
            symbolic_values=self.build_kwargs,
            config=self.config,
        )
        sample_shape = self.input_tensors["sample"]._attrs["shape"]
        down_block_additional_residuals = []
        shape = [
            sample_shape[0],
            sample_shape[1],
            sample_shape[2],
            self.config["block_out_channels"][0],
        ]
        name = "down_block_additional_residual_0"
        down_block_additional_residual = Tensor(
            shape,
            name=name,
            is_input=True,
            is_optional=True,
            dtype=self.config["dtype"],
        )
        down_block_additional_residuals.append(down_block_additional_residual)
        down_block_additional_residual_idx = 1
        for idx, block_out_channel in enumerate(self.config["block_out_channels"]):
            if idx < len(self.config["block_out_channels"]) - 1:
                # with downsample
                num_samples = self.config["layers_per_block"] + 1
            else:
                # no downsample
                num_samples = self.config["layers_per_block"]
            for sample_idx in range(num_samples):
                is_downsample = sample_idx == self.config["layers_per_block"]
                shape = [
                    sample_shape[0],
                    sample_shape[1] / pow(2, idx),
                    sample_shape[2] / pow(2, idx),
                    block_out_channel,
                ]
                if is_downsample:
                    shape[1] = shape[1] / 2
                    shape[2] = shape[2] / 2
                name = f"down_block_additional_residual_{down_block_additional_residual_idx}"
                down_block_additional_residual = Tensor(
                    shape,
                    name=name,
                    is_input=True,
                    is_optional=True,
                    dtype=self.config["dtype"],
                )
                down_block_additional_residuals.append(down_block_additional_residual)
                down_block_additional_residual_idx += 1
        shape = [
            sample_shape[0],
            sample_shape[1] / (len(self.config["block_out_channels"]) * 2),
            sample_shape[2] / (len(self.config["block_out_channels"]) * 2),
            self.config["block_out_channels"][-1],
        ]
        mid_block_additional_residual = Tensor(
            shape,
            name="mid_block_additional_residual",
            is_input=True,
            is_optional=True,
            dtype=self.config["dtype"],
        )
        self.input_tensors["down_block_additional_residuals"] = (
            down_block_additional_residuals
        )
        self.input_tensors["mid_block_additional_residual"] = (
            mid_block_additional_residual
        )
        batch = list(self.input_tensors.values())[0]._attrs["shape"][0]
        for name, tensor in self.input_tensors.items():
            if isinstance(tensor, dict):
                for sub_name, sub_tensor in tensor.items():
                    sub_tensor._attrs["shape"][0] = batch
                    print(f"{sub_name=}: {get_shape(sub_tensor)} {sub_tensor.dtype()}")
            elif isinstance(tensor, list):
                for idx, sub_tensor in enumerate(tensor):
                    sub_tensor._attrs["shape"][0] = batch
                    print(
                        f"{name=}[{idx}] ({sub_tensor._attrs['name']}): {get_shape(sub_tensor)} {sub_tensor.dtype()}"
                    )
            else:
                print(f"{name=}: {get_shape(tensor)} {tensor.dtype()}")
                tensor._attrs["shape"][0] = batch
