"""G-Eval Metric Implementations."""

from typing import List, Tuple

import torch
import torchmetrics
from typing_extensions import override

from eva.language.metrics.llm_judge.g_eval.judge import GEvalJudge
from eva.language.models import wrappers
from eva.language.models.typings import PredictionBatch


class GEvalCorrectness(torchmetrics.Metric):
    """A version of the G-Eval metric focusing on answer correctness."""

    _score_range: Tuple[int, int] = (0, 3)
    """The score range for the G-Eval judge."""

    _score_explanation: str = (
        "and is calculated as the sum of three binary criteria: "
        "Factuality, Completeness, and Consistency with the Ground Truth"
    )

    _evaluation_steps: List[str] = [
        (
            "Read the Model Response and Ground Truth carefully, "
            "and assess the following three criteria:\n"
            "- Factuality: Is the Model Response factually correct?\n"
            "- Completeness: Is the Model Response complete?\n"
            "- Consistency: Are all facts in the Model Response "
            "consistent with the Ground Truth?"
        ),
        (
            "For each of the above three criteria assign 1 point "
            "if the criterion is met, else 0 points."
        ),
    ]
    """The evaluation steps to be used by the G-Eval judge."""

    total: torch.Tensor
    count: torch.Tensor

    def __init__(self, model: wrappers.LanguageModel | str | None):
        """Initializes the metric.

        Args:
            model: An instance of the language model to use, or the
                name of the model to load from the registry.
        """
        super().__init__()
        self.judge = GEvalJudge(
            model=model,
            evaluation_steps=self._evaluation_steps,
            score_range=self._score_range,
            score_explanation=self._score_explanation,
        )

        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    @override
    @torch.no_grad()
    def update(self, preds: List[str], targets: List[str]):
        batch = PredictionBatch(prediction=preds, target=targets, text=None, metadata=None)
        scores = self.judge.evaluate(batch)

        scores_t = torch.as_tensor(scores, dtype=torch.float, device=self.device)
        self.total += scores_t.sum()
        self.count += scores_t.numel()

    @override
    def compute(self) -> torch.Tensor:
        return self.total / self.count
