try:
    import torch.distributed as dist
except (ImportError, AttributeError):
    dist = None


def is_distributed() -> bool:
    """Check if current environment is distributed.

    Returns:
        bool: True if distributed environment (e.g. multiple gpu processes).
    """
    return (
        dist is not None
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    )
