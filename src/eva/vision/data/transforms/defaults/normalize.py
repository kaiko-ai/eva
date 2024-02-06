"""Default normalization data."""

from eva.vision.data.transforms import structs

DEFAULT_IMAGE_NORMALIZATION = structs.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
"""Default image normalization."""
