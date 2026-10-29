import json
import os
import numpy as np
import torch

ARCH_NAMES = {"vit_large": "vitl", "vit_base": "vitb", "vit_small": "vits"}


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def short_name(model_name):
    """e.g. vit_large_patch16_dinov3.sat493m -> vitl-sat, convnext_tiny.dinov3_lvd1689m -> convnext_tiny-web."""
    arch, _, tag = model_name.partition(".")
    arch = next((short for long, short in ARCH_NAMES.items() if arch.startswith(long)), arch)
    return "%s-%s" % (arch, "sat" if "sat" in tag else "web")


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
