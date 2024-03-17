"""Binary balanced accuracy metric."""

from torch import Tensor
from torchmetrics.classification import stat_scores
from torchmetrics.utilities.compute import _safe_divide


class BinaryBalancedAccuracy(stat_scores.BinaryStatScores):
    """Computes the balanced accuracy for binary classification."""

    is_differentiable: bool = False
    higher_is_better: bool | None = True
    full_state_update: bool = False
    plot_lower_bound: float | None = 0.0
    plot_upper_bound: float | None = 1.0

    def compute(self) -> Tensor:
        """Compute accuracy based on inputs passed in to ``update`` previously."""
        tp, fp, tn, fn = self._final_state()
        sensitivity = _safe_divide(tp, tp + fn)
        specificity = _safe_divide(tn, tn + fp)
        return 0.5 * (sensitivity + specificity)
