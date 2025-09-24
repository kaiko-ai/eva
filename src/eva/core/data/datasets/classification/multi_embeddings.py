"""Dataset class for where a classification task sample corresponds to multiple embeddings."""

import numpy as np

from eva.core.data.datasets.multi_embeddings import MultiEmbeddingsDataset


class MultiEmbeddingsClassificationDataset(MultiEmbeddingsDataset):
    """Dataset class for where a sample corresponds to multiple embeddings.

    Specialised for classification data with an int target type.
    """

    def __init__(self, *args, **kwargs):
        """Initialize dataset with the correct return type."""
        super().__init__(*args, target_type=np.int64, **kwargs)
