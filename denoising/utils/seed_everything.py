import os
import random
from typing import Any
import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Seed `random`, `numpy`, and `torch` (CPU & CUDA) RNGs for reproducibility.

    Args:
        seed (int): The seed to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behaviour
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed) 