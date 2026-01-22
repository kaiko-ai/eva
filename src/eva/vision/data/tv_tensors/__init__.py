"""Custom `tv_tensors` types for torchvision."""

import torch
from torchvision import tv_tensors

from eva.vision.data.tv_tensors.volume import Volume

__all__ = ["Volume"]

# Monkey-patch tv_tensors.wrap to support Volume
_original_wrap = tv_tensors.wrap


@torch.compiler.disable
def _patched_wrap(
    wrappee: torch.Tensor, *, like: tv_tensors.TVTensor, **kwargs
) -> tv_tensors.TVTensor:
    """Patched version of tv_tensors.wrap that supports Volume."""
    if isinstance(like, Volume):
        return Volume._wrap(
            wrappee,
            affine=kwargs.get("affine", like.affine),
            metadata=kwargs.get("metadata", like.metadata),
        )
    return _original_wrap(wrappee, like=like, **kwargs)


tv_tensors.wrap = _patched_wrap
