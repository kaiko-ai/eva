"""Dataset class for loading pre-generated text predictions."""

import abc
from pathlib import Path
from typing import Any, Callable, Dict, Generic

import pandas as pd
from loguru import logger
from typing_extensions import override

from eva.language.data.datasets.base import LanguageDataset
from eva.language.data.datasets.schemas import TransformsSchema
from eva.language.data.datasets.typings import PredictionSample, TargetType


class TextPredictionDataset(
    LanguageDataset[PredictionSample[TargetType]], abc.ABC, Generic[TargetType]
):
    """Dataset class for loading pre-generated text predictions."""

    def __init__(
        self,
        path: str,
        prediction_column: str = "prediction",
        target_column: str = "target",
        metadata_columns: list[str] | None = None,
        pre_transforms: Callable | None = None,
        transforms: TransformsSchema | None = None,
    ):
        """Initialize the dataset.

        Args:
            path: The path to the manifest file holding the predictions & targets.
            prediction_column: The name of the prediction column.
            target_column: The name of the label column.
            metadata_columns: List of column names to include in metadata.
            pre_transforms: A callable to apply to each row in the dataframe loaded from
                the provided .parquet file in the `prepare_data` method. Note that this
                method should either accept and return a pandas Series or a dictionary.
            transforms: The transforms to apply to the text and target when
                loading the samples.
        """
        super().__init__()

        self.path = path
        self.prediction_column = prediction_column
        self.target_column = target_column
        self.metadata_columns = metadata_columns
        self.pre_transforms = pre_transforms
        self.transforms = transforms

        self._data: pd.DataFrame

    @override
    def setup(self) -> None:
        extension = Path(self.path).suffix

        match extension:
            case ".jsonl":
                self._data = pd.read_json(self.path, lines=True)
            case ".csv":
                self._data = pd.read_csv(self.path)
            case ".parquet":
                self._data = pd.read_parquet(self.path)
            case _:
                raise ValueError(f"Unsupported file extension: {extension}")

        if self.pre_transforms is not None:
            logger.info("Applying dataset pre-transforms ...")
            self._data = pd.DataFrame(self._data.apply(self._apply_pre_transform, axis=1))

        self._validate_columns()

    def _validate_columns(self) -> None:
        """Validates that required columns exist in the dataframe."""
        if self.prediction_column not in self._data.columns:
            raise ValueError(f"Label column '{self.prediction_column}' not found.")
        if self.target_column not in self._data.columns:
            raise ValueError(f"Label column '{self.target_column}' not found.")
        if self.metadata_columns:
            missing_columns = set(self.metadata_columns) - set(self._data.columns)
            if missing_columns:
                raise ValueError(f"Metadata columns {missing_columns} not found.")

    def _apply_pre_transform(self, row: pd.Series) -> pd.Series:
        if self.pre_transforms:
            try:
                row = self.pre_transforms(row)
            except Exception:
                row = self.pre_transforms(row.to_dict())
        return pd.Series(row)

    @override
    def __len__(self) -> int:
        return len(self._data)

    @override
    def __getitem__(self, index: int) -> PredictionSample[TargetType]:
        item = PredictionSample(
            prediction=self.load_prediction(index),
            target=self.load_target(index),
            metadata=self.load_metadata(index) or {},
        )
        return self._apply_transforms(item)

    def load_metadata(self, index: int) -> Dict[str, Any] | None:
        """Returns the metadata for the given index."""
        if self.metadata_columns is None:
            return None

        row = self._data.iloc[index]
        return {col: row[col] for col in self.metadata_columns}

    def load_target(self, index: int) -> TargetType:
        """Returns the target for the given index."""
        return self._data.iloc[index][self.target_column]

    def load_prediction(self, index: int) -> TargetType:
        """Returns the prediction for the given index."""
        return self._data.iloc[index][self.prediction_column]

    def _apply_transforms(
        self, sample: PredictionSample[TargetType]
    ) -> PredictionSample[TargetType]:
        """Applies the dataset transforms to the prediction and target."""
        if self.transforms:
            prediction = (
                self.transforms.prediction(sample.prediction)
                if self.transforms.prediction
                else sample.prediction
            )
            target = (
                self.transforms.target(sample.target) if self.transforms.target else sample.target
            )
            return PredictionSample(
                prediction=prediction,
                target=target,
                metadata=sample.metadata,
            )
        return sample
