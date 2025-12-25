import click

from honey.builder.t5 import T5EncoderBuilder

"""
python scripts/t5_build.py --hf-hub hlky/t5-v1_1-xxl-encoder --label xxl --batch-size 1 --min-seq 8 --max-seq 512
"""


@click.command()
@click.option(
    "--hf-hub",
    type=str,
    required=True,
)
@click.option(
    "--label",
    type=str,
    required=True,
)
@click.option(
    "--batch-size",
    type=int,
    default=1,
)
@click.option(
    "--min-seq",
    type=int,
    default=8,
)
@click.option(
    "--max-seq",
    type=int,
    default=512,
)
@click.option(
    "--dtype",
    type=str,
    default="bfloat16",
)
def build(
    hf_hub: str,
    label: str,
    batch_size: int,
    min_seq: int,
    max_seq: int,
    dtype: str = "bfloat16",
    device: str = "cuda",
):
    variant = None
    if dtype == "bfloat16":
        variant = "bf16"
    elif dtype == "float16":
        variant = "fp16"
    builder = T5EncoderBuilder(
        hf_hub=hf_hub,
        label=label,
        dtype=dtype,
        device=device,
        build_kwargs={
            "batch_size": (1, batch_size),
            "sequence_length": (min_seq, max_seq),
        },
        model_kwargs={
            "variant": variant,
        },
    )
    builder()


if __name__ == "__main__":
    build()
