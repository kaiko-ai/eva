"""Language Dataloaders API."""

from eva.language.data.dataloaders.collate_fn import prediction_collate, text_collate

__all__ = ["text_collate", "prediction_collate"]
