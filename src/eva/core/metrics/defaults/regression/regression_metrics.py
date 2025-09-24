"""Default metric collection for regression tasks."""

from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score

from eva.core.metrics import structs


class RegressionMetrics(structs.MetricCollection):
    """Default metrics for regression tasks.

    Supports:
         Mean Absolute Error
         Root Mean Squared Error
         R^2 score
    """

    def __init__(
        self,
        prefix: str | None = None,
        postfix: str | None = None,
    ) -> None:
        """Initialises regression metrics.

        Args:
            prefix: A string to prepend to metric names.
            postfix: A string to append after metric names.
        """
        super().__init__(
            metrics={
                "MAE": MeanAbsoluteError(),
                "RMSE": MeanSquaredError(squared=False),
                "R2": R2Score(),
            },
            prefix=prefix,
            postfix=postfix,
            compute_groups=[["MAE", "RMSE", "R2"]],
        )
