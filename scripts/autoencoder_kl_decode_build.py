import click
import os
# os.environ["DISABLE_PROFILER_CODEGEN"] = "1"
# os.environ["CI_FLAG"] = "CIRCLECI"
os.environ["DINOML_ALLOCATION_MODE"] = "2"

from dinoml.builder.autoencoder_kl import AutoencoderKLDecodeBuilder

"""
python scripts/autoencoder_kl_decode_build.py --hf-hub runwayml/stable-diffusion-v1-5 --subfolder vae --label v1 --batch-size 1 --min-res 8 --max-res 512
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
def build(
    hf_hub: str,
    label: str,
    batch_size: int,
    min_res: int,
    max_res: int,
    subfolder: str,
    dtype: str = "float16",
    device: str = "cuda",
):
    builder = AutoencoderKLDecodeBuilder(
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
    )
    builder()


if __name__ == "__main__":
    build()
