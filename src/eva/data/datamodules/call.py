"""Helper dataset calling methods."""
from typing import Iterable

from eva.data import datasets as datasets_lib
from eva.data.datamodules import schemas


def call_method(datasets: schemas.DatasetsSchema, method: str) -> None:
    """Calls dataset `method` from the datasets which have implemented it.

    Args:
        datasets: The datasets schema to call the method.
        method: The dataset method name to call if exists.
    """
    for dataset in _datasets_iterator(datasets):
        if hasattr(dataset, method):
            fn = getattr(dataset, method)
            fn()


def _datasets_iterator(datasets: schemas.DatasetsSchema) -> Iterable[datasets_lib.Dataset]:
    """Iterates thought the datasets.

    Args:
        datasets: The datasets to iterate from.

    Yields:
        The individual dataset class.
    """
    data_splits = [
        datasets.train,
        datasets.val,
        datasets.test,
        datasets.predict,
    ]
    data_splits = [dataset for dataset in data_splits if dataset is not None]
    for dataset in data_splits:
        if not isinstance(dataset, list):
            dataset = [dataset]
        yield from dataset
