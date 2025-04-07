from honey.builder.unet2d_condition import UNet2DConditionBuilder

builder = UNet2DConditionBuilder(
    hf_hub="runwayml/stable-diffusion-v1-5",
    label="v1",
    dtype="float16",
    device="cuda",
    build_kwargs={
        "batch_size": (1, 2),
        "resolution": (64, 512),
        "seq_len": 77,
    },
    model_kwargs={
        "subfolder": "unet",
    }
)
builder()
