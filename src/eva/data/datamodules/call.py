"""Helper dataset calling methods."""

import itertools
from typing import Iterable

from eva.data import datasets as datasets_lib
from eva.data.datamodules import schemas


def call_method_if_exists(
    datasets: schemas.DatasetsSchema, method: str, stage: str | None = None
) -> None:
    """Calls dataset `method` from the datasets if exists.

    Args:
        datasets: The datasets schema to call the method.
        method: The dataset method name to call if exists.
        stage: The stage from which the method is called. If None, this method
            will assume that the stage was not available or relevant in the context
            calling this method.
    """
    for dataset in _datasets_iterator(datasets, stage):
        if hasattr(dataset, method):
            fn = getattr(dataset, method)
            fn()


def _datasets_iterator(
    datasets: schemas.DatasetsSchema, stage: str | None = None
) -> Iterable[datasets_lib.Dataset]:
    """Iterates thought the defined datasets in a schema.

    Args:
        datasets: The datasets to iterate from.
        stage: The stage to iterate the datasets (fit, test, predict). If None, all
            available datasets are loaded.

    Yields:
        The individual dataset class.
    """
    stage_to_data_splits = {
        "fit": [datasets.train, datasets.val],
        "validate": [datasets.val],
        "test": [datasets.test],
        "predict": [datasets.predict],
    }

    if stage is None:
        data_splits = list(itertools.chain.from_iterable(stage_to_data_splits.values()))
    elif stage not in stage_to_data_splits:
        raise ValueError(f"Valid stages are: fit, test, predict. Received {stage}.")
    else:
        data_splits = stage_to_data_splits[stage]
    data_splits = [dataset for dataset in data_splits if dataset is not None]

    for dataset in data_splits:
        if not isinstance(dataset, list):
            dataset = [dataset]
        yield from dataset
