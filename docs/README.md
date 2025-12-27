# Getting started

## Install

```sh
git clone --recursive https://github.com/hlky/dinoml
cd dinoml
pip install -e .
```

### Dependencies

Some modeling is based on [🧨 Diffusers](https://github.com/huggingface/diffusers) and [Transformers](https://github.com/huggingface/transformers). We use `diffusers` and `transformers` to load the original model weights when building those DinoML modules.

```sh
pip install diffusers
pip install transformers==4.57.3
```

## Next steps

[Building modules](./MODULES.md)

[Modeling](./MODELING.md)

[Environment variables](./ENVIRONMENT.md)