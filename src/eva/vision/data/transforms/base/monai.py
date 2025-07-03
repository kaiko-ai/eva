"""Base class for MONAI transform wrappers."""

import abc

from torchvision.transforms import v2


class RandomMonaiTransform(v2.Transform, abc.ABC):
    """Base class for MONAI transform wrappers."""

    @abc.abstractmethod
    def set_random_state(self, seed: int) -> None:
        """Set the random state for the transform.

        MONAI's random transforms use numpy.random for random number generation
        which is seeded at the very beginning by lightning's seed_everything, but when
        torch spins up the dataloader workers, it will only reseed torch's random states
        and not numpy - so you basically end up with multiple dataloader workers that
        have equally seeded random transforms, resulting in redundant transform outputs
        and therefore reducing the diversity of the resulting training data.
        To solve this, this method should be called in the dataloader's worker_init_fn
        with a unique seed for each worker.

        Args:
            seed: The seed to set for the random state of the transform.
        """
        pass
