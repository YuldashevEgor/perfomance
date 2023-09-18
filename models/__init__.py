import torch
import timm
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
import torch.optim.lr_scheduler as lr_scheduler

from typing import List


def get_seg_model(model_type, params):
    return getattr(smp, model_type)(**params)


def get_model(cfg: dict, task):
    if task == "seg":
        model = get_seg_model(cfg['model_type'], cfg['hparams'])
    elif task == "cls":
        model = timm.create_model(cfg['model_type'],
                                  num_classes=cfg['hparams']["classes"],
                                  pretrained=False)
    else:
        raise AttributeError
    return model


def get_optimizer(cfg: dict,
                  biases: List[torch.Tensor],
                  not_biases: List[torch.Tensor]) -> torch.optim:
    optimizer = getattr(optim, cfg['type'])(params=[{'params': biases, 'lr': 2 * cfg["params"]["lr"]},
                                                    {'params': not_biases}],
                                            **cfg['params'])
    return optimizer


def get_scheduler(cfg: dict,
                  optimizer: optim) -> lr_scheduler:
    scheduler = getattr(lr_scheduler, cfg['type'])(optimizer,
                                                   **cfg['params'])
    return scheduler


def get_criterion(cfg: dict):
    criterion = getattr(nn, cfg['type'])(**cfg['params'])
    return criterion
