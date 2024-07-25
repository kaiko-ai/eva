from typing import List, Tuple

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from transformers.models.mask2former import modeling_mask2former


class Mask2formerMatcher(nn.Module):
    def __init__(
        self,
        num_points: int,
        mask_coefficient: float,
        dice_coefficient: float,
        class_coefficient: float | None = None,
    ) -> None:
        super().__init__()
        self.num_points = num_points
        self.mask_coefficient = mask_coefficient
        self.dice_coefficient = dice_coefficient
        self.class_coefficient = class_coefficient

    def create_cost_matrix(
        self,
        pred_mask,
        target_mask,
        cost_class=None,
    ):
        point_coordinates = torch.rand(1, self.num_points, 2, device=pred_mask.device)
        target_coordinates = point_coordinates.repeat(target_mask.shape[0], 1, 1)
        target_mask = modeling_mask2former.sample_point(
            target_mask, target_coordinates, align_corners=False
        ).squeeze(1)

        pred_coordinates = point_coordinates.repeat(pred_mask.shape[0], 1, 1)
        pred_mask = modeling_mask2former.sample_point(
            pred_mask, pred_coordinates, align_corners=False
        ).squeeze(1)

        cost_mask = modeling_mask2former.pair_wise_sigmoid_cross_entropy_loss(
            pred_mask, target_mask
        )
        cost_dice = modeling_mask2former.pair_wise_dice_loss(pred_mask, target_mask)

        cost_matrix = self.mask_coefficient * cost_mask + self.dice_coefficient * cost_dice
        if cost_class is not None:
            cost_matrix += self.class_coefficient * cost_class

        return cost_matrix

    @torch.no_grad()
    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        mask_labels: List[torch.Tensor],
        class_queries_logits: torch.Tensor | None = None,
        class_labels: List[torch.Tensor] | None = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        indices_list = []
        batch_size = masks_queries_logits.shape[0]

        for i in range(batch_size):
            pred_mask = masks_queries_logits[i]
            target_mask = mask_labels[i]
            target_mask, pred_mask = target_mask[:, None], pred_mask[:, None]

            cost_class = None
            if class_queries_logits is not None and class_labels is not None:
                pred_probs = class_queries_logits[i].softmax(-1)
                cost_class = -pred_probs[:, class_labels[i]]

            cost_matrix = self.create_cost_matrix(pred_mask, target_mask, cost_class)
            cost_matrix = torch.minimum(cost_matrix, torch.tensor(1e10))
            cost_matrix = torch.maximum(cost_matrix, torch.tensor(-1e10))
            i_indices, j_indices = linear_sum_assignment(cost_matrix.cpu())

            indices_list.append((i_indices, j_indices))

        return [
            (
                torch.as_tensor(i, dtype=torch.int, device=masks_queries_logits.device),
                torch.as_tensor(j, dtype=torch.int, device=masks_queries_logits.device),
            )
            for i, j in indices_list
        ]
