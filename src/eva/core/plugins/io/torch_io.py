"""Extended version of the IO model checkpoint plugin `TorchCheckpointIO`."""

import os
from typing import Any, Dict

from lightning.fabric.utilities import cloud_io
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch import plugins
from typing_extensions import override


class SubmoduleTorchCheckpointIO(plugins.TorchCheckpointIO):
    """IO plugin which allows to additionally save only a sub-part of the full model."""

    def __init__(self, submodule: str) -> None:
        """Initializes the plugin.

        Args:
            submodule: The name of the submodule to additionally save.
        """
        super().__init__()

        self._submodule = submodule

    @override
    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path: _PATH,
        storage_options: Any | None = None,
    ) -> None:
        super().save_checkpoint(checkpoint, path, storage_options)
        self._save_submodule(checkpoint["state_dict"], path)

    @override
    def remove_checkpoint(self, path: _PATH) -> None:
        super().remove_checkpoint(path)
        self._remove_submodule(path)

    def _save_submodule(self, module_checkpoint: Dict[str, Any], module_path: _PATH) -> None:
        """Saves the submodule."""
        path = self._submodule_path(module_path)
        state_dict = self._submodule_state_dict(module_checkpoint)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        cloud_io._atomic_save(state_dict, path)

    def _remove_submodule(self, module_path: _PATH) -> None:
        """Removes the submodule."""
        path = self._submodule_path(module_path)
        fs = cloud_io.get_filesystem(path)
        if fs.exists(path):
            fs.rm(path, recursive=True)

    def _submodule_path(self, module_path: _PATH) -> str:
        """Constructs and returns the submodule checkpoint path."""
        root, basename = os.path.split(module_path)
        return os.path.join(root, self._submodule, basename.replace(".ckpt", ".pth"))

    def _submodule_state_dict(self, module_state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Returns the submodule `state_dict`."""
        key = self._submodule if self._submodule.endswith(".") else self._submodule + "."
        return {
            module.replace(key, ""): weights
            for module, weights in module_state_dict.items()
            if module.startswith(key)
        }
