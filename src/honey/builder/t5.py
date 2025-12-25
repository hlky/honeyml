from honey.builder.base import Build
from honey.mapping.t5 import map_t5
from honey.utils.build_utils import build_tensors_from_annotations
from honey.utils.shape_utils import get_shape


class T5EncoderBuilder(Build):
    """

    Example:
    ```

    builder = T5EncoderBuilder(
        hf_hub="hlky/t5-v1_1-xxl-encoder",
        label="v1",
        dtype="bfloat16",
        device="cuda",
        build_kwargs={
            "batch_size": 1,
            "sequence_length": (8, 512),
        },
        model_kwargs={
            "variant": "bf16",
        }
    )

    ```

    """

    model_name = "t5.{model_type}.{label}.{device_name}.sm{sm}"
    model_type = "encoder"
    map_function = map_t5
    model_output_names = ["last_hidden_state"]
    model_output = "last_hidden_state"

    def create_input_tensors(self):
        build_kwargs = self.build_kwargs.copy()
        build_kwargs["num_heads"] = self.config["num_heads"]
        build_kwargs.update({"input_ids": {"dtype": "int64"}})
        self.input_tensors = build_tensors_from_annotations(
            getattr(self.honey_module, self.model_forward),
            symbolic_values=build_kwargs,
            config={"input_ids": {"dtype": "int64"}},
        )
        batch = list(self.input_tensors.values())[0]._attrs["shape"][0]
        for name, tensor in self.input_tensors.items():
            if isinstance(tensor, dict):
                for sub_name, sub_tensor in tensor.items():
                    sub_tensor._attrs["shape"][0] = batch
                    print(f"{sub_name=}: {get_shape(sub_tensor)} {sub_tensor.dtype()=}")
            else:
                # TODO
                print(f"{name=}: {get_shape(tensor)} {tensor.dtype()=}")
                tensor._attrs["shape"][0] = batch
