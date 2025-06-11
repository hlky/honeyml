import torch
from honey.compiler import Model

device = "cuda"
dtype = torch.float16

path = "./tmp/autoencoder_kl.decode.xl.1024.3090.sm86/autoencoder_kl.decode.xl.1024.3090.sm86.so"

module = Model(path)

min_res = 1
max_res = 128

for res in range(min_res, max_res + 1):
    inputs = {
        "z": torch.randn([1, 4, res, res], dtype=dtype)
        .permute(0, 2, 3, 1)
        .contiguous()
        .to(device)
    }
    outputs = {
        "Y": torch.empty([1, 3, res * 8, res * 8], dtype=dtype)
        .permute(0, 2, 3, 1)
        .contiguous()
        .to(device)
    }
    mean, std, _ = module.benchmark_with_tensors(
        inputs=inputs,
        outputs=outputs,
        count=50,
        repeat=3,
    )
    print(f"[{res=}] {mean=:.3f}, {std=:.3f}")
