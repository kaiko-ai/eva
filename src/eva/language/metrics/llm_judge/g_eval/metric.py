from typing import List

import torchmetrics


class GEval(torchmetrics.Metric):
    """A placeholder for the G-Eval metric implementation."""

    def __init__(self):
        super().__init__()

    def update(self, preds: List[str], targets: List[str]):
        pass

    def compute(self):
        pass
