"""Checkpointing related utilities and helper functions."""

from typing import Any, Dict


def submodule_state_dict(state_dict: Dict[str, Any], submodule_key: str) -> Dict[str, Any]:
    """Get the state dict of a specific submodule.

    Args:
        state_dict: The state dict to extract the submodule from.
        submodule_key: The key of the submodule to extract.

    Returns:
        The subset of the state dict corresponding to the specified submodule.
    """
    submodule_key = submodule_key if submodule_key.endswith(".") else submodule_key + "."
    return {
        module: weights
        for module, weights in state_dict.items()
        if module.startswith(submodule_key)
    }
