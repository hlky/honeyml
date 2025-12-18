from typing import List
import click

from honey.builder.autoencoder_kl import AutoencoderKLEncodeBuilder

"""
python scripts/autoencoder_kl_encode_build.py --hf-hub runwayml/stable-diffusion-v1-5 --subfolder vae --label v1 --batch-size 1 --min-res 8 --max-res 512
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
    "--subfolder",
    type=str,
    default=None,
)
@click.option(
    "--dtype",
    type=str,
    default="float16",
)
@click.option(
    "--sample-mode",
    multiple=True,
    default=["sample"],
)
def build(
    hf_hub: str,
    label: str,
    batch_size: int,
    min_res: int,
    max_res: int,
    subfolder: str,
    sample_mode: List[str],
    dtype: str = "float16",
    device: str = "cuda",
):
    builder = AutoencoderKLEncodeBuilder(
        hf_hub=hf_hub,
        label=label,
        dtype=dtype,
        device=device,
        build_kwargs={
            "batch_size": (1, batch_size),
            "resolution": (min_res, max_res),
        },
        model_kwargs={
            "subfolder": subfolder,
        },
        forward_kwargs={
            "sample_mode": sample_mode,
        },
    )
    builder()


if __name__ == "__main__":
    build()
