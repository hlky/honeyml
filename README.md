# DinoML

<div align="center"><img src="assets/dinoml.png" width="128"></div>

**DinoML** compiles machine learning models into optimized standalone *modules*.

## System and platform support

| System  | Support |
| ------- |:-------:|
| Windows | ✅ |
| Linux   | ✅ |

| Platform | Support |
| -------- |:-------:|
| CUDA     | ✅ |
| ROCm     | Partial |

## Optimizations

- Custom kernels for standard operators
- Custom fused kernels that combine several operations into a single kernel
- Graph transformations deduplicate and reduce the number of needed operations
- Graph transformations fuse common patterns of operations
- Profiling to find the best kernel for a problem size
- Intermediate workspace is reused to reduce memory usage

and more!

## Comparison

|        | Dynamic shape | Portability | Dynamic memory usage |
|--------|:-------------:|:-----------:|:--------------------:|
| DinoML | ✅ | ✅ | ✅ |
| torch.compile | Limited | Limited | ❌ |
| TensorRT | Limited | Limited | ❌ |

## Documentation

- [Getting started](docs/README.md)
- [Modules](docs/MODULES.md)
- [Modeling](docs/MODELING.md)
