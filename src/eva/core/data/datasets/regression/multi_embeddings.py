"""Dataset class for where a regression task sample corresponds to multiple embeddings."""

import numpy as np

from eva.core.data.datasets.multi_embeddings import MultiEmbeddingsDataset


class MultiEmbeddingsRegressionDataset(MultiEmbeddingsDataset):
    """Dataset class for where a sample corresponds to multiple embeddings.

    Specialised for regression data with a float target type.
    """

    def __init__(self, *args, **kwargs):
        """Initialize dataset with the correct return type."""
        super().__init__(*args, target_type=np.float32, **kwargs)
