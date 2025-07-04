import click

from honey.builder.esrgan import ESRGANBuilder

"""
python scripts/esrgan_build.py --hf-hub hlky/RealESRGAN_x4plus --label x4plus --batch-size 1 --min-res 8 --max-res 512
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
    "--min-res",
    type=int,
    default=8,
)
@click.option(
    "--max-res",
    type=int,
    default=512,
)
@click.option(
    "--dtype",
    type=str,
    default="float16",
)
def build(
    hf_hub: str,
    label: str,
    batch_size: int,
    min_res: int,
    max_res: int,
    dtype: str = "float16",
    device: str = "cuda",
):
    builder = ESRGANBuilder(
        hf_hub=hf_hub,
        label=label,
        dtype=dtype,
        device=device,
        build_kwargs={
            "batch_size": (1, batch_size),
            "resolution": (min_res, max_res),
        },
    )
    builder()


if __name__ == "__main__":
    build()
