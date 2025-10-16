"""G-Eval Metric Implementations."""

from typing import List, Tuple

import torch
import torchmetrics
from torch import nn
from typing_extensions import override

from eva.language.metrics.llm_judge.g_eval.judge import GEvalJudge
from eva.language.models import wrappers
from eva.language.models.typings import PredictionBatch


class GEvalCorrectness(torchmetrics.Metric):
    """A version of the G-Eval metric focusing on answer correctness."""

    _score_range: Tuple[int, int] = (1, 5)
    """The score range for the G-Eval judge."""

    _evaluation_steps: List[str] = [
        "Read the Model Response and Ground Truth carefully",
        (
            "Identify Key Facts: Extract all important facts, claims, and information "
            "from the Ground Truth response."
        ),
        (
            "Assess Correctness & Completeness: For each key fact in the Ground Truth, "
            "determine if it appears in the Model Response (exactly or paraphrased), "
            "and evaluate whether all essential information from Ground Truth is present."
        ),
        (
            "Identify Errors: Note any factual contradictions or inaccuracies in the "
            "Model Response compared to Ground Truth."
        ),
    ]
    """The evaluation steps to be used by the G-Eval judge."""

    _scoring_criteria: str = "\n".join(
        [
            (
                "5 (Excellent): Model Response captures all key facts from Ground Truth "
                "accurately. Information is complete and correct, with no factual errors or "
                "contradictions. May use different wording but conveys equivalent meaning."
            ),
            (
                "4 (Good): Model Response captures most key facts correctly with no significant "
                "errors. May miss 1-2 minor details, but all major points are present and accurate."
            ),
            (
                "3 (Acceptable): Model Response captures about half of the key information "
                "accurately. Some important facts are missing, or there are minor inaccuracies, "
                "but no major contradictions with Ground Truth."
            ),
            (
                "2 (Poor): Model Response captures only a small portion of key facts. Major "
                "information is missing and may contain factual errors or contradictions with "
                "Ground Truth."
            ),
            (
                "1 (Very Poor): Model Response is largely incorrect or incomplete, missing "
                "most key facts. Contains significant factual errors or contradictions with "
                "Ground Truth."
            ),
        ]
    )

    total: torch.Tensor
    count: torch.Tensor

    def __init__(self, model: wrappers.LanguageModel | nn.Module | str | None):
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
            scoring_criteria=self._scoring_criteria,
        )

        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    @override
    @torch.no_grad()
    def update(self, preds: List[str], targets: List[str]):
        batch = PredictionBatch(prediction=preds, target=targets, text=None, metadata=None)
        scores = self.judge.evaluate(batch)
        # TODO: filter & count None scores

        scores_t = torch.as_tensor(scores, dtype=torch.float, device=self.device)
        self.total += scores_t.sum()
        self.count += scores_t.numel()

    @override
    def compute(self) -> torch.Tensor:
        return self.total / self.count
