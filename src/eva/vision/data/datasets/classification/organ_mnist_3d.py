"""OrganMNIST3D dataset wrapper."""

import os
from typing import Callable

import medmnist
from medmnist.info import DEFAULT_ROOT


class OrganMNIST3D(medmnist.OrganMNIST3D):
    """Abdominal CT Multi-Class (11) classification dataset.

    More info:
      - MedMNIST
        https://medmnist.com/
    """

    def __init__(
        self,
        split: str,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
        as_rgb: bool = False,
        root: str | None = DEFAULT_ROOT,
        size: int | None = 64,
        mmap_mode: str | bool | None = None,
    ) -> None:
        """Initialize the OrganMNIST3D dataset.

        Args:
            split: Dataset split, e.g., 'train', 'val', or 'test'.
            transform: Optional transform to apply to the images.
            target_transform: Optional transform to apply to the labels.
            download: Whether to download the dataset if not found.
            as_rgb: Convert images to RGB if True.
            root: Root directory to store/load the dataset.
            size: Resize images to this size.
            mmap_mode: Memory mapping mode, or None.
        """
        if root is not None:
            os.makedirs(root, exist_ok=True)

        super().__init__(
            split=split,
            transform=transform,
            target_transform=target_transform,
            download=download,
            as_rgb=as_rgb,
            root=root,
            size=size,
            mmap_mode=mmap_mode,
        )
