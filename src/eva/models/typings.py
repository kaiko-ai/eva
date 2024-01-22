"""Type annotations for model modules."""
from typing import Union

import pytorch_lightning as pl
from torch import nn

ModelType = Union[nn.Module, pl.LightningModule]
"""The expected model type."""
