from typing import Union

import pytorch_lightning as pl
from torch import nn


ModelType = Union[nn.Module, pl.LightningModule]
"""The model type."""
