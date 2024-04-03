"""Sampling transformations."""

import torch


class SampleFromAxis:
    """Samples n_samples entries from a tensor along a given axis."""

    def __init__(self, n_samples: int, seed: int = 42, axis: int = 0):
        """Initialize the transformation.

        Args:
            n_samples: The number of samples to draw.
            seed: The seed to use for sampling.
            axis: The axis along which to sample.
        """
        self._seed = seed
        self._n_samples = n_samples
        self._axis = axis
        self._generator = self._get_generator()

    def _get_generator(self):
        """Return a torch random generator with fixed seed."""
        generator = torch.Generator()
        generator.manual_seed(self._seed)
        return generator

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Call method for the transformation.

        Args:
            tensor: The input tensor of shape [n, embedding_dim].

        Returns:
            A tensor of shape [n_samples, embedding_dim].
        """
        indices = torch.randperm(tensor.size(self._axis), generator=self._generator)[
            : self._n_samples
        ]
        return tensor.index_select(self._axis, indices)
