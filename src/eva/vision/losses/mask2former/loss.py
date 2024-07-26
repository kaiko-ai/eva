from typing import List

import torch
import torch.nn as nn
from torch import distributed
from transformers.models.mask2former import modeling_mask2former

from eva.vision.losses.mask2former import matcher


class Mask2formerLoss(modeling_mask2former.Mask2FormerLoss):
    def __init__(
        self,
        num_labels: int,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        class_coefficient: float | None = 2.0,
        no_object_coefficient: float | None = 0.1,
    ) -> None:
        nn.Module.__init__(self)

        self.num_labels = num_labels
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.mask_coefficient = mask_coefficient
        self.dice_coefficient = dice_coefficient
        self.class_coefficient = class_coefficient
        self.no_object_coefficient = self.eos_coef = no_object_coefficient

        if self.num_labels is not None:
            empty_weight = torch.ones(self.num_labels + 1)
            empty_weight[-1] = self.eos_coef  # type: ignore
            self.register_buffer("empty_weight", empty_weight)

        self.matcher = matcher.Mask2formerMatcher(
            num_points=num_points,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
        )

    @torch.compiler.disable
    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        mask_labels: List[torch.Tensor],
        class_queries_logits: torch.Tensor | None = None,
        class_labels: List[torch.Tensor] | None = None,
    ):
        """Smthing.

        Args:
            masks_queries_logits: A tensor of shape `(batch_size, num_queries, height, width)`.
            mask_labels: List of mask labels of shape `(labels, height, width)`.
            class_queries_logits: A tensor of shape `(batch_size, num_queries, num_labels)`.
            class_labels: List of class labels of shape `(labels)`.
        """
        mask_labels = [mask.float() for mask in mask_labels]
        class_labels = [label.long() for label in class_labels]

        indices = self.matcher(
            masks_queries_logits=masks_queries_logits,
            mask_labels=mask_labels,
            class_queries_logits=class_queries_logits,
            class_labels=class_labels,
        )

        loss_masks = self.loss_masks(masks_queries_logits, mask_labels, indices, 1)
        num_masks = sum(len(tgt) for (_, tgt) in indices)
        num_masks_tensor = torch.as_tensor(
            num_masks, dtype=torch.float, device=masks_queries_logits.device
        )

        if distributed.is_available() and distributed.is_initialized():
            distributed.all_reduce(num_masks_tensor)
            world_size = distributed.get_world_size()
        else:
            world_size = 1

        num_masks = torch.clamp(num_masks_tensor / world_size, min=1).item()

        for key in loss_masks.keys():
            loss_masks[key] = loss_masks[key] / num_masks

        loss_classes = self.loss_labels(class_queries_logits, class_labels, indices)  # type: ignore

        return {**loss_masks, **loss_classes}

    def loss_total(self, losses_all_layers, log_fn) -> torch.Tensor:
        loss_total = None
        for loss_key, loss in losses_all_layers.items():
            log_fn(f"train_{loss_key}", loss, sync_dist=True)

            if "mask" in loss_key:
                weighted_loss = loss * self.mask_coefficient
            elif "dice" in loss_key:
                weighted_loss = loss * self.dice_coefficient
            elif "cross_entropy" in loss_key:
                weighted_loss = loss * self.class_coefficient
            else:
                raise ValueError(f"Unknown loss key: {loss_key}")

            if loss_total is None:
                loss_total = weighted_loss
            else:
                loss_total = torch.add(loss_total, weighted_loss)

        log_fn("train_loss_total", loss_total, sync_dist=True, prog_bar=True)

        return loss_total  # type: ignore
