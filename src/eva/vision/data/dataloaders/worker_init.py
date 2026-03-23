"""Dataloader worker init functions."""

import random

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms.v2

from eva.vision.data.transforms import base


def seed_worker(worker_id: int) -> None:
    """Sets the random seed for each dataloader worker process.

    How to use?
    `torch.utils.data.Dataloader(..., worker_init_fn=seed_worker)`

    Args:
        worker_id: The ID of the worker process.
    """
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None and hasattr(worker_info, "dataset"):
        dataset = torch.utils.data.get_worker_info().dataset  # type: ignore
        if hasattr(dataset, "_transforms"):
            transforms = dataset._transforms  # type: ignore
            if isinstance(transforms, torchvision.transforms.v2.Compose):
                for transform in transforms.transforms:
                    if isinstance(transform, base.RandomMonaiTransform):
                        transform.set_random_state(seed=worker_seed)
