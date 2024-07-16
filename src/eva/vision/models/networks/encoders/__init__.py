"""Encoder networks API."""

from eva.vision.models.networks.encoders.encoder import Encoder
from eva.vision.models.networks.encoders.from_timm import TimmEncoder
from eva.vision.models.networks.encoders.phikon import PhikonEncoder
from eva.vision.models.networks.encoders.wrapper import EncoderWrapper

__all__ = ["Encoder", "TimmEncoder", "PhikonEncoder", "EncoderWrapper"]
