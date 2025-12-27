from dinoml.builder.transformer_flux import FluxTransformer2DBuilder

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
    },
    store_constants_in_module=False,
    benchmark_after_compile=False,
)
builder()
