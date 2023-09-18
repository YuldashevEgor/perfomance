import torchvision.transforms as transforms
import torchvision
import torch

from torchvision.transforms.functional import InterpolationMode
from enum import Enum


class TaskType(str, Enum):
    cls = "cls"
    seg = "seg"


def get_cls_dataloader(split: str,
                       transform):
    dataset = torchvision.datasets.CIFAR10(root='./data/cls',
                                           train=split == "train",
                                           download=True,
                                           transform=transform)

    return dataset


def replace_tensor_value_(tensor, a, b):
    tensor[tensor == a] = b
    return tensor


def get_seg_dataloader(split: str,
                       transform,
                       input_size):
    target_transform = transforms.Compose(
        [
            transforms.Resize(input_size, interpolation=InterpolationMode.NEAREST),
            transforms.PILToTensor(),
            transforms.Lambda(lambda x: replace_tensor_value_(x.squeeze(0).long(), 255, 21)),
        ]
    )
    dataset = torchvision.datasets.VOCSegmentation(root='./data/seg',
                                                   image_set=split,
                                                   download=True,
                                                   transform=transform,
                                                   target_transform=target_transform)

    return dataset


def get_dataset_and_dataloader(split: str,
                               cfg):
    transform = transforms.Compose(
        [transforms.Resize(cfg["input_size"]),
         transforms.ToTensor()])

    task = cfg["task"]
    if task == TaskType.cls:
        dataset = get_cls_dataloader(split, transform)
    elif task == TaskType.seg:
        dataset = get_seg_dataloader(split, transform, cfg["input_size"])
    else:
        raise AttributeError

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=cfg["batch_size"],
                                             shuffle=split == "train",
                                             num_workers=cfg["num_workers"])
    return dataset, dataloader
