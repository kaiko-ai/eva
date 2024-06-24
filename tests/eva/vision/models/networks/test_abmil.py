"""ABMIL network tests."""

import itertools
from typing import Tuple

import pytest
import torch

from eva.vision.models.networks import ABMIL


@pytest.mark.parametrize(
    "input_size, output_size, hidden_sizes_mlp, batch_size, n_instances, masked_fraction",
    list(itertools.product([50, 384], [6], [(), (128, 64)], [1, 16], [100], [0.1, 0.6])),
)
def test_masked_abmil(
    input_size: int,
    output_size: int,
    hidden_sizes_mlp: Tuple[int],
    batch_size: int,
    n_instances: int,
    masked_fraction: float,
) -> None:
    """Test if abmil model yields same output in masked and unmasked case."""
    pad_value = float("-inf")
    model = ABMIL(
        input_size=input_size,
        output_size=output_size,
        projected_input_size=128,
        hidden_size_attention=128,
        hidden_sizes_mlp=hidden_sizes_mlp,
        use_bias=True,
        pad_value=None,
    )

    n_masked = int(n_instances * masked_fraction)

    x = torch.randn(batch_size, n_instances, input_size)
    x[:, n_masked:, :] = pad_value

    # without padding
    y = model(x[:, :n_masked, :])

    # with padding
    model._pad_value = pad_value
    y_masked = model(x)

    assert torch.allclose(y, y_masked, atol=1e-6)
