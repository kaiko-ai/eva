"""Learning Rate scheduler configuration."""

import dataclasses
from typing import Any, Literal

from lightning.pytorch.cli import LRSchedulerCallable
from torch import optim


@dataclasses.dataclass
class SchedulerConfiguration:
    """Initializes and builds the learning rate scheduler configuration."""

    scheduler: LRSchedulerCallable
    """The learning rate scheduler instance."""

    interval: Literal["step", "epoch"] = "epoch"
    """The unit of the scheduler's step size.

    It can be 'step' or 'epoch', to update the scheduler on step or epoch end respectively.
    """

    frequency: int = 1
    """How many epochs/steps should pass between calls to `scheduler.step()`.

    Value `1` corresponds to updating the learning rate after every epoch/step.
    """

    monitor: str = "val_loss"
    """Metric to to monitor for schedulers like `ReduceLROnPlateau`."""

    strict: bool = True
    """Whether to enforce that the value specified 'monitor' must be available.

    If the values is not available when the scheduler is updated it will stop the
    training. With `False`, it will only produce a warning.
    """

    name: str | None = None
    """Specifies a custom logged name for the `LearningRateMonitor` callback."""

    def __call__(self, optimizer: optim.Optimizer) -> dict[str, Any]:
        """Returns Lightning's lr_scheduler_config configuration."""
        return {
            "scheduler": self.scheduler(optimizer),
            "interval": self.interval,
            "frequency": self.frequency,
            "monitor": self.monitor,
            "strict": self.strict,
            "name": self.name,
        }
