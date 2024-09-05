"""ViT Adapter module."""

import functools
import math
from typing import Callable, List

import timm.models.layers
import timm.models.vision_transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from timm.models.layers import trunc_normal_
from torch.nn.init import normal_
from typing_extensions import override

import eva.core.models
from eva.vision.models.networks.adapters.vit_adapter._adapter_modules import (
    InteractionBlock,
    SpatialPriorModule,
    deform_inputs,
)
from eva.vision.models.networks.adapters.vit_adapter._ms_deform_attn import MSDeformAttn


class ViTAdapter(nn.Module):
    """ViT Adapter module."""

    def __init__(
        self,
        vit_backbone: timm.models.vision_transformer.VisionTransformer | eva.core.models.BaseModel,
        interaction_indexes: List[List[int]] | None = None,
        pretrain_size=224,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=6,
        init_values=0.0,
        with_cffn=True,
        cffn_ratio=0.25,
        deform_ratio=1.0,
        add_vit_feature=True,
        use_extra_extractor=True,
        freeze_vit=True,
        norm_layer=functools.partial(nn.LayerNorm, eps=1e-6),
    ):
        """Initializes the ViTAdapter."""
        super().__init__()

        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        self.freeze_vit = freeze_vit
        self.norm_layer = norm_layer
        self.cls_token = None
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.vit_backbone = self._get_vit_backbone(vit_backbone)

        self._setup_model()
        embed_dim = self.vit_backbone.embed_dim
        self.drop_path_rate = self._get_drop_path_rate()
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False)
        self.interactions = nn.Sequential(
            *[
                InteractionBlock(
                    dim=embed_dim,
                    num_heads=deform_num_heads,
                    n_points=n_points,
                    init_values=init_values,
                    drop_path=self.drop_path_rate,
                    norm_layer=self.norm_layer,
                    with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio,
                    deform_ratio=deform_ratio,
                    extra_extractor=(
                        (True if i == len(self.interaction_indexes) - 1 else False)
                        and use_extra_extractor
                    ),
                    with_cp=False,
                )
                for i in range(len(self.interaction_indexes))
            ]
        )
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    @functools.cached_property
    def patch_size(self) -> int:
        patch_size = self.vit_backbone.patch_embed.patch_size  # type: ignore
        if patch_size[0] != patch_size[1]:
            raise ValueError(f"Patch size must be square, got {patch_size}.")
        return patch_size[0]

    def _setup_model(self) -> None:
        self._verify_norm_layer(self.norm_layer)
        if self.freeze_vit:
            self._freeze_backbone()
        self._calculate_interactions()

    def _get_vit_backbone(
        self, vit_backbone: nn.Module
    ) -> timm.models.vision_transformer.VisionTransformer:
        if isinstance(vit_backbone, eva.core.models.BaseModel):
            while hasattr(vit_backbone, "_model"):
                vit_backbone = vit_backbone._model
        if not isinstance(vit_backbone, timm.models.vision_transformer.VisionTransformer):
            raise ValueError(
                f"ViTAdapter only supports timm.VisionTransformer got {type(vit_backbone)}."
            )
        return vit_backbone

    def _calculate_interactions(self, n_interactions: int = 4) -> List[List[int]]:
        if self.interaction_indexes is None:
            n_blocks = len(self.vit_backbone.blocks)
            if n_blocks % n_interactions != 0:
                raise ValueError(f"Number of vit blocks must be divisible by {n_interactions}.")
            block_size = n_blocks // n_interactions
            start_indexes = list(range(0, n_blocks, block_size))
            self.interaction_indexes = [
                [start_indexes[i], start_indexes[i] + block_size - 1]
                for i in range(len(start_indexes))
            ]
            logger.info(f"Using interactions at {self.interaction_indexes} for {n_blocks} blocks.")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed: torch.Tensor, H: int, W: int, patch_size: int):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // patch_size, self.pretrain_size[0] // patch_size, -1
        ).permute(0, 3, 1, 2)
        pos_embed = (
            F.interpolate(pos_embed, size=(H, W), mode="bicubic", align_corners=False)
            .reshape(1, -1, H * W)
            .permute(0, 2, 1)
        )
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def _verify_norm_layer(self, norm_layer: Callable):
        """Check if the norm layer type matches the one used in the VisionTransformer."""
        norm_layer_type = (
            norm_layer.func if isinstance(norm_layer, functools.partial) else norm_layer
        )
        if type(self.vit_backbone.norm) != norm_layer_type:  # type: ignore
            raise ValueError(
                f"norm_layer type mismatch: {type(self.vit_backbone.norm)} != {norm_layer_type}"
            )

    def _freeze_backbone(self):
        if self.freeze_vit:
            for param in self.vit_backbone.parameters():
                param.requires_grad = False

    def _get_drop_path_rate(self):
        drop_path = self.vit_backbone.blocks[-1].drop_path1
        return drop_path.drop_prob if isinstance(drop_path, timm.models.layers.DropPath) else 0.0

    @override
    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x, self.patch_size)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        x = self.vit_backbone.patch_embed(x)
        if x.dim() == 4:  # NCHW -> NLC
            x = x.reshape(x.shape[0], -1, x.shape[3])
        H = W = int(math.sqrt(x.shape[1]))
        bs, n, dim = x.shape
        pos_embed = self._get_pos_embed(self.vit_backbone.pos_embed[:, 1:], H, W, self.patch_size)
        x = self.vit_backbone.pos_drop(x + pos_embed)

        # Interaction
        H2, W2 = 14, 14  # 1/16 * 224
        outs = []
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(
                x,
                c,
                self.vit_backbone.blocks[indexes[0] : indexes[-1] + 1],
                deform_inputs1,
                deform_inputs2,
                H2,
                W2,
            )
            outs.append(x.transpose(1, 2).view(bs, dim, H, W).contiguous())

        # Split & Reshape
        c2 = c[:, 0 : c2.size(1), :]
        c3 = c[:, c2.size(1) : c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1) :, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H2 * 2, W2 * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H2, W2).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H2 // 2, W2 // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            interpolate = functools.partial(F.interpolate, mode="bilinear", align_corners=False)
            x1 = interpolate(x1, scale_factor=c1.shape[2] / x1.shape[2])
            x2 = interpolate(x2, scale_factor=c2.shape[2] / x2.shape[2])
            x3 = interpolate(x3, scale_factor=c3.shape[2] / x3.shape[2])
            x4 = interpolate(x4, scale_factor=c4.shape[2] / x4.shape[2])
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)  # 1/4
        f2 = self.norm2(c2)  # 1/8
        f3 = self.norm3(c3)  # 1/16
        f4 = self.norm4(c4)  # 1/32

        return [f1, f2, f3]
