import os  
import json  
from typing import List, Tuple  
import torch.nn as nn 

# Poner en 0 pesos y bias de capas conv2D y batchnorm2d
def remove_filters(model: nn.Module, filters: List[Tuple[str, int]]) -> None:
    for layer_name, filter_index in filters:
        parts = layer_name.split('.')
        module = model
        for part in parts[:-1]:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        layer = getattr(module, parts[-1])
        if isinstance(layer, nn.Conv2d):
            layer.weight.data[int(filter_index)].fill_(0)
            if layer.bias is not None:
                layer.bias.data[int(filter_index)] = 0
        elif isinstance(layer, nn.BatchNorm2d):
            layer.weight.data[int(filter_index)] = 0
            layer.bias.data[int(filter_index)] = 0

def get_filters(model: nn.Module) -> List[Tuple[str, int]]:
    filters = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            filters.extend([(f"{name}.weight", i) for i in range(module.out_channels)])
        elif isinstance(module, nn.BatchNorm2d):
            filters.extend([(f"{name}.weight", i) for i in range(module.num_features)])
    return filters

# Checkpoints
def save_checkpoint(state, sample_number):
    # Create checkpoint filename with sample number
    filename = f"{CHECKPOINT_BASE}_sample_{sample_number}.json"
    checkpoint_path = os.path.join(MODEL_PATH, filename)

    # Create checkpoint directory if it doesn't exist
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    with open(checkpoint_path, 'w') as f:
        json.dump(state, f, indent=4)
    print(f"Checkpoint saved to {checkpoint_path}")

def load_checkpoint(sample_number=None):
    if sample_number is None:
        # Find the latest checkpoint
        checkpoints = [f for f in os.listdir(MODEL_PATH) if f.startswith(CHECKPOINT_BASE) and f.endswith('.json')]
        if not checkpoints:
            raise FileNotFoundError("No checkpoints found")
        
        # Extract sample numbers and get the latest
        samples = [int(f.split('_sample_')[1].split('.json')[0]) for f in checkpoints]
        latest_sample = max(samples)
        filename = f"{CHECKPOINT_BASE}_sample_{latest_sample}.json"
    else:
        filename = f"{CHECKPOINT_BASE}_sample_{sample_number}.json"
    
    checkpoint_path = os.path.join(MODEL_PATH, filename)
    
    with open(checkpoint_path, 'r') as f:
        state = json.load(f)
    print(f"Checkpoint loaded from {checkpoint_path}")
    return state
