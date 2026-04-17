# Getting started

## Install

```sh
git clone --recursive https://github.com/hlky/dinoml
cd dinoml
pip install torch
pip install -e .
```

### Dependencies

The base package metadata is intentionally small and does not install GPU runtime packages for you. For actual module building you should install PyTorch first, then the model-loading stack used by DinoML builders.

Some modeling is based on [🧨 Diffusers](https://github.com/huggingface/diffusers) and [Transformers](https://github.com/huggingface/transformers). DinoML uses them to load source configs and weights before converting them into DinoML constants.

```sh
pip install diffusers
pip install transformers==4.57.3
```

## Next steps

[Building modules](./MODULES.md)

[Modeling](./MODELING.md)

[Architecture](./ARCHITECTURE.md)

[Profiling](./PROFILING.md)

[Environment variables](./ENVIRONMENT.md)
