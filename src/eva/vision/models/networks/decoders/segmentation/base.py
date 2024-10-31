"""Semantic segmentation decoder base class."""

import abc

import torch
from torch import nn

from eva.vision.models.networks.decoders.segmentation.typings import DecoderInputs


class Decoder(nn.Module, abc.ABC):
    """Abstract base class for segmentation decoders."""

    @abc.abstractmethod
    def forward(self, decoder_inputs: DecoderInputs) -> torch.Tensor:
        """Forward pass of the decoder."""
