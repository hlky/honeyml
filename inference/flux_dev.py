from typing import List

import torch
import tqdm

from honey.compiler import Model
from honey.compiler.dtype import _ENUM_TO_TORCH_DTYPE

module_out_path = "H:/honey_modules/flux-dev"
device = "cuda"


def ProcessBlock(block_name, modules, max_blocks):
    inputs = {}
    outputs = {}
    block_idx = 0
    pbar = tqdm.tqdm(desc=block_name, total=max_blocks)
    for module in modules:
        if block_idx == 0:
            for name, idx in module.get_input_name_to_index_map().items():
                shape = module.get_input_maximum_shape(idx)
                dtype = _ENUM_TO_TORCH_DTYPE[module.get_input_dtype(idx)]
                tensor = torch.randn(*shape, dtype=dtype).to(device)
                inputs[name] = tensor
        else:
            inputs["hidden_states"] = outputs["hidden_states_out"]
            if "encoder_hidden_states" in inputs:
                inputs["encoder_hidden_states"] = outputs["encoder_hidden_states_out"]
        if block_idx == 0:
            for name, idx in module.get_output_name_to_index_map().items():
                shape = module.get_output_maximum_shape(idx)
                dtype = _ENUM_TO_TORCH_DTYPE[module.get_output_dtype(idx)]
                tensor = torch.empty(*shape, dtype=dtype).to(device)
                outputs[name] = tensor
        module.run_with_tensors(inputs=inputs, outputs=outputs)
        block_idx += 1
        pbar.update(1)
    for idx in range(len(modules), max_blocks):
        module = Model(f"{module_out_path}/{block_name}.{idx}.dll")
        module.run_with_tensors(inputs=inputs, outputs=outputs)
        inputs["hidden_states"] = outputs["hidden_states_out"]
        if "encoder_hidden_states" in inputs:
            inputs["encoder_hidden_states"] = outputs["encoder_hidden_states_out"]
        pbar.update(1)
    return outputs


transformer_block_modules: List[Model] = []
for idx in range(0, 19):
    module = Model(f"{module_out_path}/FluxTransformerBlock.{idx}.dll")
    transformer_block_modules.append(module)
single_transformer_block_modules: List[Model] = []
for idx in range(0, 27):
    module = Model(f"{module_out_path}/FluxSingleTransformerBlock.{idx}.dll")
    single_transformer_block_modules.append(module)

for _ in tqdm.tqdm(range(20), desc="steps"):
    _ = ProcessBlock("FluxTransformerBlock", transformer_block_modules, 19)
    _ = ProcessBlock("FluxSingleTransformerBlock", single_transformer_block_modules, 38)
