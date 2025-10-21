"""Utility functions for distributed training."""

import torch.distributed as dist


def is_distributed() -> bool:
    """Check if current environment is distributed.

    Returns:
        bool: True if distributed environment (e.g. multiple gpu processes).
    """
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def is_main_process() -> bool:
    """Check if a process is the main process in distributed training.

    Returns:
        bool: True if main process, i.e., rank 0.
    """
    return dist.get_rank() == 0 if dist.is_initialized() else True
