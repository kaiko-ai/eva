from typing import Callable, Tuple

import pytest
import torch
import torchmetrics


def _test_ignore_index(
    metric_cls: Callable[..., torchmetrics.Metric],
    batch_size: int,
    num_classes: int,
    image_size: Tuple[int, int],
    ignore_index: int,
) -> None:
    """Test ignore index functionality of a torchmetric."""
    generator = torch.Generator()
    generator.manual_seed(42)

    metric = metric_cls(num_classes=num_classes)
    preds = torch.randint(0, num_classes, (batch_size,) + image_size, generator=generator)
    target = preds.clone()
    result_one = metric(preds=preds, target=target)

    random_mask = torch.randint(0, 2, (batch_size,) + image_size).bool()
    preds[random_mask] = ignore_index  # simulate wrong predictions
    result_two = metric(preds=preds, target=target)

    metric_with_ignore = metric_cls(num_classes=num_classes, ignore_index=ignore_index)
    result_three = metric_with_ignore(preds=preds, target=target)

    assert result_one != pytest.approx(result_two, abs=1e-6)
    assert result_one == pytest.approx(result_three, abs=1e-6)
