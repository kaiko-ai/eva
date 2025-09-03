"""Collate functions API."""

from eva.language.data.dataloaders.collate_fn.text import prediction_collate, text_collate

__all__ = ["text_collate", "prediction_collate"]
