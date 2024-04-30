"""Encoder networks API."""

from eva.vision.models.networks.encoders.encoder import Encoder
from eva.vision.models.networks.encoders.from_timm import TimmEncoder

__all__ = ["Encoder", "TimmEncoder"]
