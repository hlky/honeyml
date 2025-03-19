import torch

from honey.compiler import Model

import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed

device = "cuda"

graph_mode = False
sync = False

def _decode(inputs, outputs):
    module = Model("./autoencoder_kl.l4.sm89.so")
    return module.run_with_tensors(
        inputs=inputs,
        outputs=outputs,
        sync=sync,
        graph_mode=graph_mode,
    )

with ThreadPoolExecutor(8) as executor:
    futures = [executor.submit(_decode, {
            "z": torch.randn([1, 4, 64, 64], dtype=torch.float16)
            .permute(0, 2, 3, 1)
            .contiguous()
            .to(device)
        },
        {
            "Y": torch.empty([1, 3, 512, 512], dtype=torch.float16)
            .permute(0, 2, 3, 1)
            .contiguous()
            .to(device)
        })]
    for future in tqdm.tqdm(as_completed(futures)):
        result = future.result()

