"""The Mask2Former criterion."""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch import distributed
from transformers.models.mask2former import modeling_mask2former

from eva.vision.losses.mask2former import matcher


class Mask2FormerLoss(modeling_mask2former.Mask2FormerLoss):
    """The Mask2Former criterion."""

    def __init__(
        self,
        num_classes: int,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        class_coefficient: float = 2.0,
        no_object_coefficient: float | None = 0.1,
    ) -> None:
        """Initialize the class.

        Args:
            num_classes: Number of classes in the dataset.
            num_points: Number of points to sample for the loss computation.
            oversample_ratio: Ratio for oversampling during point sampling.
            importance_sample_ratio: Ratio for importance sampling.
            mask_coefficient: Coefficient for the mask loss.
            dice_coefficient: Coefficient for the dice loss.
            class_coefficient: Coefficient for the class loss.
            no_object_coefficient: Coefficient for the no-object loss.
        """
        nn.Module.__init__(self)

        self.num_classes = self.num_labels = num_classes
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.mask_coefficient = mask_coefficient
        self.dice_coefficient = dice_coefficient
        self.class_coefficient = class_coefficient
        self.no_object_coefficient = self.eos_coef = no_object_coefficient

        if self.num_classes is not None:
            empty_weight = torch.ones(self.num_classes + 1)
            empty_weight[-1] = self.eos_coef  # type: ignore
            self.register_buffer("empty_weight", empty_weight)

        self.matcher = matcher.Mask2FormerMatcher(
            num_points=num_points,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
        )

    def _compute_layer_loss(
        self,
        masks_queries_logits: torch.Tensor,
        mask_labels: List[torch.Tensor],
        class_queries_logits: torch.Tensor,
        class_labels: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Computes the losses of a single layer.

        Args:
            masks_queries_logits: A tensor of shape `(batch_size, num_queries, height, width)`.
            mask_labels: List of mask labels of shape `(labels, height, width)`.
            class_queries_logits: A tensor of shape `(batch_size, num_queries, num_classes)`.
            class_labels: List of class labels of shape `(labels)`.

        Returns:
            The dictionary containing the computed losses.
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

    def _compute_total_loss(
        self,
        masks_queries_logits: List[torch.Tensor],
        mask_labels: List[torch.Tensor],
        class_queries_logits: List[torch.Tensor],
        class_labels: List[torch.Tensor],
    ) -> torch.Tensor:
        """Computes the aggregated total loss of all the layers.

        Args:
            masks_queries_logits: List of tensors of shape
                `(batch_size, num_queries, height, width)`.
            mask_labels: List of mask labels of shape `(labels, height, width)`.
            class_queries_logits: List of tensors of shape `(batch_size, num_queries, num_classes)`.
            class_labels: List of class labels of shape `(labels)`.

        Returns:
            The computed total loss.
        """
        total_loss = torch.tensor(0.0, device=masks_queries_logits[0].device)
        for masks_queries, class_queries in zip(
            masks_queries_logits, class_queries_logits, strict=True
        ):
            loss = self._compute_layer_loss(masks_queries, mask_labels, class_queries, class_labels)
            weighted_loss = (
                loss.get("loss_mask", torch.tensor(0.0, device=masks_queries.device))
                * self.mask_coefficient
                + loss.get("loss_dice", torch.tensor(0.0, device=masks_queries.device))
                * self.dice_coefficient
                + loss.get("loss_cross_entropy", torch.tensor(0.0, device=masks_queries.device))
                * self.class_coefficient
            )
            total_loss = torch.add(total_loss, weighted_loss)

        return total_loss

    @torch.compiler.disable
    def forward(
        self,
        inputs: Tuple[List[torch.Tensor], List[torch.Tensor]],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the total loss.

        Args:
            inputs: A tuple with the mask and class logits of shape
                `(batch_size, num_queries, height, width)` and
                `(batch_size, num_queries, num_classes)` respectively.
            targets: The target semantic mask labels.

        Returns:
            The computed total loss.
        """
        masks_queries_logits, class_queries_logits = inputs
        mask_labels, class_labels = _convert_semantic_label(targets)
        return self._compute_total_loss(
            masks_queries_logits=masks_queries_logits,
            mask_labels=mask_labels,
            class_queries_logits=class_queries_logits,
            class_labels=class_labels,
        )


def _convert_semantic_label(
    semantic_labels: torch.Tensor,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Converts semantic labels to mask and class labels.

    Args:
        semantic_labels: Tensor of shape `(batch_size, height, width)`
            representing the semantic labels.

    Returns:
        A tuple containing the list of mask labels and list of class labels.
    """
    mask_labels, class_labels = [], []
    for batch in semantic_labels:
        masks, labels = [], []
        for label_id in batch.unique():
            masks.append((batch == label_id))
            labels.append(label_id)
        mask_labels.append(torch.stack(masks))
        class_labels.append(torch.stack(labels))
    return mask_labels, class_labels
