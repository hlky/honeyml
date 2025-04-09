from honey.builder.autoencoder_kl import AutoencoderKLDecodeBuilder

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
builder()
