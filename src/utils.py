"""Small helpers: seeding, (de)normalisation, heat-map overlays, environment info."""
import json
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def denormalize(x: torch.Tensor) -> np.ndarray:
    """(3,H,W) ImageNet-normalised tensor -> uint8 (H,W,3) image."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device)[:, None, None]
    std = torch.tensor(IMAGENET_STD, device=x.device)[:, None, None]
    img = (x * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def overlay(img_u8: np.ndarray, heat: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a [0,1] heat-map (jet colour map) over a uint8 image; returns float RGB in [0,1]."""
    import matplotlib
    colored = matplotlib.colormaps["jet"](heat)[..., :3]
    return np.clip((1 - alpha) * img_u8 / 255.0 + alpha * colored, 0, 1)


def env_info() -> dict:
    import torchvision
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
    return info


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
