from dinoml.builder.t2i_adapter import T2IAdapterBuilder

builder = T2IAdapterBuilder(
    hf_hub="hlky/t2iadapter_canny_sd15v2",
    label="canny_sd15v2",
    dtype="float16",
    device="cuda",
    build_kwargs={
        "batch_size": (1, 2),
        "resolution": (8, 512),
    },
    check_outputs=False,
)
builder()
