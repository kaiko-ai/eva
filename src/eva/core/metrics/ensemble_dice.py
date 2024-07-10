# """Ensemble Dice (DICE2) metric."""

# import torch
# from loguru import logger
# from typing_extensions import override

# from eva.core.metrics import structs


# class DICE2(structs.Metric):
#     """Computes the ensemble dice score for nuclear instance segmentation."""

#     is_differentiable: bool = False
#     higher_is_better: bool | None = True
#     full_state_update: bool = False
#     plot_lower_bound: float | None = 0.0
#     plot_upper_bound: float | None = 1.0

#     def __init__(self) -> None:
#         """Initializes the metric."""
#         super().__init__()

#         self.add_state("value", default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

#     @override
#     def update(self, loss: torch.Tensor) -> None:
#         _check_nans(loss)
#         total_samples = loss.numel()
#         if total_samples == 0:
#             return

#         self.value = self.value + torch.sum(loss)
#         self.total = self.total + total_samples

#     @override
#     def compute(self) -> torch.Tensor:
#         return self.value / self.total
