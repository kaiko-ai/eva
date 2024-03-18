"""Datasets API."""

from eva.data.datasets import classification
from eva.data.datasets.base import Dataset
from eva.data.datasets.dataset import TorchDataset

__all__ = ["classification", "Dataset", "TorchDataset"]
