"""Text prediction writer callbacks."""

from collections import deque

from eva.language.callbacks.loggers.diagnostic import (
    DiagnosticLoggerCallback as BaseDiagnosticLoggerCallback,
)


class DiagnosticLoggerCallback(BaseDiagnosticLoggerCallback):
    """Callback for logging diagnostic information during training and evaluation."""

    def __init__(
        self,
        log_generations: bool = True,
        log_sample_size: int = 100,
    ) -> None:
        """Initializes a new callback.

        Args:
            log_generations: Whether to log the generated text & samplewise metrics.
            log_sample_size: The number of samples to log if `log_generations` is True
        """
        super().__init__(log_generations, log_sample_size)

        if self.log_generations:
            self._data["objects"] = deque(maxlen=log_sample_size)
