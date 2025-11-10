"""
Utility functions: seeding, saving checkpoints, simple logger helpers.
"""

import os
import random
import json
from pathlib import Path
import numpy as np
import torch

from . import config


def set_seed(seed: int = None):
    seed = seed or config.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs():
    Path(config.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.RESULTS_DIR).mkdir(parents=True, exist_ok=True)


def save_checkpoint(state: dict, filename: str):
    ensure_dirs()
    torch.save(state, os.path.join(config.CHECKPOINT_DIR, filename))


def load_checkpoint(path: str, model: torch.nn.Module = None, optimizer: torch.optim.Optimizer = None):
    ckpt = torch.load(path, map_location="cpu")
    if model is not None:
        model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
