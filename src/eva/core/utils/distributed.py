"""Utility functions for distributed training."""

import torch.distributed as dist


def is_distributed() -> bool:
    """Check if current environment is distributed.

    Returns:
        bool: True if distributed environment (e.g. multiple gpu processes).
    """
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
