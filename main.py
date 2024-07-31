"""Helper wrapper class for Pathology FMs."""

from typing import Any

import torch
from torch import nn


from eva.vision.models.networks.backbones.pathology._registry import PathologyModelRegistry

model = PathologyModelRegistry.load_model("kaiko_vits16")



