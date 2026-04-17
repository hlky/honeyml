# Build

This directory contains older ad-hoc build scripts that compile specific models directly. For reusable code, prefer the builders in `src/dinoml/builder/` or the CLI wrappers in `scripts/`.

## Config

`load_config(...)`, implemented in `src/dinoml/builder/config.py`, downloads `config.json` from Hugging Face and maps the source architecture to a DinoML class through `_CLASS_MAPPING`.

Example:

```python
config, dinoml_cls, pt_cls = load_config(
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    subfolder="transformer",
)
```

`config` is passed into the DinoML module, `dinoml_cls` is the graph-building class, and `pt_cls` is used to load pretrained weights for constant mapping.

## Build Flow

The low-level scripts in this directory generally do the following:

1. Load config and resolve classes with `load_config(...)`.
2. Instantiate the DinoML module and call `name_parameter_tensor()`.
3. Create symbolic input `Tensor`s directly.
4. Run the symbolic forward pass and mark outputs with `mark_output(...)`.
5. Load a pretrained PyTorch module and convert weights into DinoML constants.
6. Call `compile_model(...)` and optionally benchmark the result.

Generated code and build artifacts are written under `tmp/<model_name>/`.

## Example Scripts

- `autoencoder_kl.py` and `autoencoder_kl_encode.py` show a manual VAE compile path.
- `sd1.py`, `sd2.py`, `sd3.py`, `sdxl.py`, and `sdxl-refiner.py` compile UNet-style diffusion backbones.
- `flux.py` and `flux-schnell.py` show transformer-based image model compilation.

If you are adding a new model family, start in `src/dinoml/builder/` unless you specifically need a one-off exploratory script.
