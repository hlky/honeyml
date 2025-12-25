"""
Verifies module works at every possible resolution combination.
"""

import torch
import json
from honey.compiler import Model

device = "cuda"
dtype = torch.float16

name = "autoencoder_kl.{type}.v1.1024.3090.sm86"

decode_path = f"./tmp/{name.format(type='decode')}/{name.format(type='decode')}.so"
encode_path = f"./tmp/{name.format(type='encode')}/{name.format(type='encode')}.so"

decode_module = Model(decode_path)
encode_module = Model(encode_path)

min_res = 1
max_res = 128
results = {"decode": [], "encode": []}

channels = 16 if "flux" in name else 4


for h in range(min_res, max_res + 1):
    for w in range(min_res, max_res + 1):
        # Decode
        inputs = {"z": torch.randn([1, h, w, channels], dtype=dtype).to(device)}
        outputs = {"Y": torch.empty([1, h * 8, w * 8, 3], dtype=dtype).to(device)}
        mean, std, _ = decode_module.benchmark_with_tensors(
            inputs=inputs, outputs=outputs, count=5, repeat=1
        )
        results["decode"].append(
            {
                "input_shape": list(inputs["z"].shape),
                "output_shape": list(outputs["Y"].shape),
                "height": h,
                "width": w,
                "mean_ms": round(mean, 3),
                "std_ms": round(std, 3),
            }
        )

        # Encode
        input_x = torch.randn([1, h * 8, w * 8, 3], dtype=dtype).to(device)
        output_shape = [1, h, w, channels]
        inputs = {
            "x": input_x,
            "sample": torch.empty(output_shape, dtype=dtype).to(device),
        }
        outputs = {"Y": torch.empty(output_shape, dtype=dtype).to(device)}
        mean, std, _ = encode_module.benchmark_with_tensors(
            inputs=inputs, outputs=outputs, count=2, repeat=1
        )
        results["encode"].append(
            {
                "input_shape": list(input_x.shape),
                "output_shape": output_shape,
                "height": h,
                "width": w,
                "mean_ms": round(mean, 3),
                "std_ms": round(std, 3),
            }
        )

with open(f"{name.format(type='decode')}_benchmark_results.json", "w") as f:
    json.dump(results["decode"], f, indent=2)

with open(f"{name.format(type='encode')}_benchmark_results.json", "w") as f:
    json.dump(results["encode"], f, indent=2)
