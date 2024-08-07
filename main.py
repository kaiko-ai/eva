"""Mask2Former from HF."""

from typing import List

import torch
from torch import nn
from transformers.models import mask2former
from transformers.models.mask2former import modeling_mask2former


class Mask2FormerModel(modeling_mask2former.Mask2FormerPreTrainedModel):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        feature_size: int = 32,
        hidden_dim: int = 32,
        num_queries: int = 32,
        decoder_layers: int = 4,
        # feature_size: int = 256,
        # hidden_dim: int = 256,
        # num_queries: int = 100,
        # decoder_layers: int = 10,
    ) -> None:
        """Mask2FormerPixelDecoderEncoderLayer initialization.

        Args:
            feature_channels: the channels of the feature maps from the encoder.
        """
        self._in_features = in_features
        self._num_classes = num_classes
        self._feature_size = feature_size
        self._hidden_dim = hidden_dim
        self._num_queries = num_queries
        self._decoder_layers = decoder_layers

        super().__init__(self._config)

        self._pixel_decoder = modeling_mask2former.Mask2FormerPixelDecoder(
            config=self._config, feature_channels=3 * [self._in_features]
        )
        self.transformer_module = modeling_mask2former.Mask2FormerTransformerModule(
            config=self._config, in_features=self._feature_size
        )
        self.class_predictor = nn.Linear(self._hidden_dim, self._num_classes + 1)

        self.post_init()

    @property
    def _config(self) -> mask2former.Mask2FormerConfig:
        return mask2former.Mask2FormerConfig(
            feature_size=self._feature_size,
            hidden_dim=self._hidden_dim,
            num_queries=self._num_queries,
            decoder_layers=self._decoder_layers,
        )

    def forward(
        self,
        features: List[torch.Tensor],
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
    ) -> modeling_mask2former.Mask2FormerModelOutput:

        pixel_level_outputs = self._pixel_decoder(
            features, output_hidden_states=output_hidden_states
        )
        transformer_module_output = self.transformer_module(
            multi_scale_features=pixel_level_outputs.multi_scale_features,
            mask_features=pixel_level_outputs.mask_features,
            output_hidden_states=True,
            output_attentions=output_attentions,
        )

        class_queries_logits = ()
        for decoder_output in transformer_module_output.intermediate_hidden_states:
            class_prediction = self.class_predictor(decoder_output.transpose(0, 1))
            class_queries_logits += (class_prediction,)

        return (
            transformer_module_output.masks_queries_logits,
            class_queries_logits,
        )


HIDDEN_FEATURES = 196
features = [
    torch.rand(4, HIDDEN_FEATURES, 64, 64),
    torch.rand(4, HIDDEN_FEATURES, 32, 32),
    torch.rand(4, HIDDEN_FEATURES, 16, 16),
]

decoder = Mask2FormerModel(in_features=HIDDEN_FEATURES, num_classes=10)
masks_queries_logits, class_queries_logits = decoder(features)
print(masks_queries_logits[0].shape)

from eva.vision.models.networks.decoders.segmentation.mask2former.head import Mask2FormerHead


