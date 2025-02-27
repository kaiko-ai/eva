"""Dataloaders API."""

from eva.core.data.dataloaders.collate_fn import text_collate_fn
from eva.core.data.dataloaders.dataloader import DataLoader

__all__ = ["DataLoader", "text_collate_fn"]
