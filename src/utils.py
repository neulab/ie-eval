import os
import sys
import shutil

from typing import NamedTuple, Optional, Tuple
from collections import OrderedDict
from datetime import datetime
from tqdm import tqdm

import torch


class Checkpoint(NamedTuple):
    epoch: int
    val_loss: float
    model_state: OrderedDict
    optim_state: dict


def cpu_state_dict(state_dict):
    res = {}
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            res[k] = v.to("cpu")
        elif isinstance(v, dict):
            res[k] = cpu_state_dict(v)
        else:
            res[k] = v
    return res


def get_best_model(path: str) -> Tuple[Optional[str], int]:
    try:
        best_epoch = max(
            (int(p[5:-3]) for p in os.listdir(path) if p.startswith("model") and p.endswith(".pt")),
            default=-1,
        )
        best_model_path = os.path.join(path, f"model{best_epoch}.pt") if best_epoch != -1 else None
        return best_model_path, best_epoch
    except FileNotFoundError:
        return None, -1


def create_exp_dir(path: str, script_path: str, overwrite=False) -> str:
    """
    Create experiment directory, and return path to log file.
    """
    if os.path.exists(path):
        if not overwrite:
            print(f"The experiment name: {path} already exists.")
            sys.exit(1)
        else:
            # Radical
            print(f"Overwriting the experiment: {path} ...")
            shutil.rmtree(path)

    os.mkdir(path)
    # shutil.copy(script_path, path)
    logfile = os.path.join(path, f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log")
    return logfile


def progress(iterator, desc=""):
    return tqdm(iterator, desc=desc, ncols=80, ascii=True)
