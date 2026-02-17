"""SegResNet-based Encoder."""

from typing import List, Tuple

import torch
from monai.inferers.inferer import Inferer
from monai.networks.blocks.segresnet_block import ResBlock, get_conv_layer
from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer
from torch import nn

from eva.vision.models.networks.backbones.registry import backbone_registry


@backbone_registry.register("segresnet_encoder")
class SegResNetEncoder(nn.Module):
    """Encoder based on the SegResNet [0].

    - [0] 3D MRI brain tumor segmentation using autoencoder regularization
      https://arxiv.org/pdf/1810.11654.pdf
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 16,
        in_channels: int = 1,
        out_indices: int | None = None,
        inferer: Inferer | None = None,
        dropout_prob: float | None = None,
        act: Tuple | str = ("RELU", {"inplace": True}),
        norm: Tuple | str = ("GROUP", {"num_groups": 8}),
        norm_name: str = "",
        num_groups: int = 8,
        blocks_down: Tuple = (1, 2, 2, 4),
    ) -> None:
        """Initialize the SegResNet encoder.

        Args:
            spatial_dims: Number of spatial dimensions (2D or 3D).
            init_filters: Number of output channels for the initial convolution layer.
            in_channels: Number of input channels for the network.
            out_indices: If set, returns intermediate features up to this index.
            inferer: An optional MONAI `Inferer` for efficient
                inference during evaluation.
            dropout_prob: Probability of an element to be zeroed out.
            act: Activation type and arguments.
            norm: Normalization type and arguments.
            norm_name: Deprecated option for feature normalization type.
            num_groups: Deprecated option for group normalization.
            blocks_down: Number of downsample blocks at each stage.
        """
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("`spatial_dims` can only be 2 or 3.")

        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(f"Deprecated option `norm_name={norm_name}`, use `norm` instead.")
            norm = ("GROUP", {"num_groups": num_groups})

        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.out_indices = out_indices
        self.inferer = inferer
        self.dropout_prob = dropout_prob
        self.act = act
        self.norm = norm
        self.blocks_down = blocks_down
        self.act_mod = get_act_layer(act)

        self.convInit = get_conv_layer(self.spatial_dims, in_channels, init_filters)
        self.down_layers = self._make_down_layers()
        self.dropout = (
            Dropout[Dropout.DROPOUT, self.spatial_dims](dropout_prob) if dropout_prob else None
        )
        self._pool_op = (
            nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
            if self.spatial_dims == 3
            else nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

        if out_indices is not None and (out_indices < 0 or out_indices > len(blocks_down)):
            raise ValueError(
                f"`out_indices` must be in range [0, {len(blocks_down)}], got {out_indices}."
            )

    def _make_down_layers(self) -> nn.ModuleList:
        """Constructs downsampling layers using ResBlocks.

        Returns:
            List of downsampling layers.
        """
        down_layers = nn.ModuleList()
        filters = self.init_filters
        for i, num_blocks in enumerate(self.blocks_down):
            in_channels = filters * (2**i)
            pre_conv = (
                get_conv_layer(self.spatial_dims, in_channels // 2, in_channels, stride=2)
                if i > 0
                else nn.Identity()
            )
            blocks = [
                ResBlock(self.spatial_dims, in_channels, norm=self.norm, act=self.act)
                for _ in range(num_blocks)
            ]
            down_layers.append(nn.Sequential(pre_conv, *blocks))
        return down_layers

    def _forward_features(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Encodes and compute intermediate feature maps through the encoder network.

        Args:
            tensor: Input tensor of shape (B, C, D, H, W) for 3D or (B, C, H, W) for 2D.

        Returns:
            A list of intermediate feature maps.
        """
        x = self.convInit(tensor)
        if self.dropout:
            x = self.dropout(x)

        intermediates = []
        for block in self.down_layers:
            x = block(x)
            intermediates.append(x)

        return intermediates

    def forward_features(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Computes feature maps using either standard forward pass or inference mode.

        If in inference mode (`self.training` is False) and an inference method
        (`self.inferer`) is available,  the `_inferer` is used to extract features.
        Otherwise, `_forward_features` is called directly.

        Args:
            tensor: Input tensor of shape (B, C, D, H, W) for 3D or (B, C, H, W) for 2D.

        Returns:
            A list of intermediate feature maps.
        """
        if not self.training and self.inferer:
            return self.inferer(inputs=tensor, network=self._forward_features)

        return self._forward_features(tensor)

    def forward_head(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Aggregate features from all encoder stages into a single feature vector.

        Args:
            features: List of feature maps from encoder stages.

        Returns:
            Aggregated feature vector of shape (B, C').
        """
        batch_size = features[0].shape[0]
        reduced_features = []
        for patch_features in features:
            hidden_features = self._pool_op(patch_features)
            hidden_features_reduced = hidden_features.view(batch_size, -1)
            reduced_features.append(hidden_features_reduced)
        return torch.cat(reduced_features, dim=1)

    def forward_embeddings(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute the final aggregated feature vector from the input tensor.

        Args:
            tensor: Input tensor of shape (B, C, T, H, W).

        Returns:
            Aggregated feature vector of shape (B, C').
        """
        features = self.forward_features(tensor)
        return self.forward_head(features)

    def forward_intermediates(
        self, tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Computes encoder features and their embeddings.

        Args:
            tensor: Input tensor of shape (B, C, T, H, W).

        Returns:
            Aggregated feature vector and list of intermediate features.
        """
        features = self.forward_features(tensor)
        embeddings = self.forward_head(features)
        return embeddings, features

    def forward(self, tensor: torch.Tensor) -> torch.Tensor | List[torch.Tensor]:
        """Forward pass through the encoder.

        If `self.out_indices` is None, it returns the aggregated feature vector.
        Otherwise, it returns the intermediate feature maps up to the specified index.

        Args:
            tensor: Input tensor of shape (B, C, T, H, W).

        Returns:
            Aggregated feature vector or intermediate features.
        """
        if self.out_indices is None:
            return self.forward_embeddings(tensor)

        intermediates = self.forward_features(tensor)
        return intermediates[-1 * self.out_indices :]
