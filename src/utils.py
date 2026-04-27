import json
import random
from pathlib import Path

import numpy as np
import torch


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preferred: str = "auto") -> torch.device:
    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred != "auto":
        raise ValueError("preferred must be one of: auto, cuda, cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_torch_runtime(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def describe_torch_runtime(device: torch.device) -> str:
    cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable"
    return (
        f"torch={torch.__version__} "
        f"cuda_available={torch.cuda.is_available()} "
        f"device={device} "
        f"cuda_device={cuda_name}"
    )


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
