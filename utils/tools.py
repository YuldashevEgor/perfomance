import os
import yaml
import torch
import random
import argparse
import numpy as np

from typing import Tuple, List


def set_random_seed(seed: int = None,
                    deterministic: bool = True) -> None:
    """
    """

    if seed is None:
        return None

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True  # noqa
        torch.backends.cudnn.benchmark = False  # noqa


def get_config() -> dict:
    """
    """
    parser = argparse.ArgumentParser(description='Read config')
    parser.add_argument("-c", "--config", required=True, help="path to config")

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.full_load(f)

    return config


def get_device(cfg: dict) -> torch.device:
    gpu_index = cfg['train']['gpu_index']

    device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() and gpu_index is not None else "cpu")

    try:
        torch.cuda.set_device(device)
    except ValueError:
        print(f"Cuda device {device} not found")

    return device


def get_biases_params(model: torch.nn.Module) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    """
    biases = list()
    not_biases = list()
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if param_name.endswith('.bias'):
                biases.append(param)
            else:
                not_biases.append(param)
    return biases, not_biases


def clean_device(device: torch.device) -> None:
    if device.type != "cpu":
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
