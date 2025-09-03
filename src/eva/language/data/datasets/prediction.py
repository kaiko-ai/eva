"""Dataset class for loading pre-generated text predictions."""

import abc
from pathlib import Path
from typing import Any, Dict, Generic, Literal

import pandas as pd
from typing_extensions import override

from eva.language.data.datasets.base import LanguageDataset
from eva.language.data.datasets.schemas import TransformsSchema
from eva.language.data.datasets.typings import PredictionSample, TargetType
from eva.language.data.messages import MessageSeries, UserMessage
from eva.language.utils.text import messages as message_utils


class TextPredictionDataset(
    LanguageDataset[PredictionSample[TargetType]], abc.ABC, Generic[TargetType]
):
    """Dataset class for loading pre-generated text predictions."""

    def __init__(
        self,
        path: str,
        prediction_column: str = "prediction",
        target_column: str = "target",
        text_column: str | None = None,
        metadata_columns: list[str] | None = None,
        split: Literal["train", "val", "test"] | None = None,
        transforms: TransformsSchema | None = None,
    ):
        """Initialize the dataset.

        Args:
            path: The path to the manifest file holding the predictions & targets.
            prediction_column: The name of the prediction column.
            target_column: The name of the label column.
            text_column: The name of the column with the text inputs that were used
                to generate the predictions. If the text column contains chat message
                json format ([{"role": ..., "content": ...}]), it will be deserialized into
                a list of Message objects. Otherwise, the content is interpreted as a
                single user message.
            metadata_columns: List of column names to include in metadata.
            split: The dataset split to use (train, val, test). If not specified,
                the entire dataset will be used.
            transforms: The transforms to apply to the text and target when
                loading the samples.
        """
        super().__init__()

        self.path = path
        self.prediction_column = prediction_column
        self.target_column = target_column
        self.text_column = text_column
        self.metadata_columns = metadata_columns
        self.split = split
        self.transforms = transforms

        self._data: pd.DataFrame

    @override
    def __len__(self) -> int:
        return len(self._data)

    @override
    def __getitem__(self, index: int) -> PredictionSample[TargetType]:
        item = PredictionSample(
            prediction=self.load_prediction(index),
            target=self.load_target(index),
            text=self.load_text(index),
            metadata=self.load_metadata(index) or {},
        )
        return self._apply_transforms(item)

    @override
    def configure(self) -> None:
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

        if self.split is not None:
            self._data = self._data[self._data["split"] == self.split].reset_index(drop=True)  # type: ignore

    @override
    def validate(self) -> None:
        if self.prediction_column not in self._data.columns:
            raise ValueError(f"Label column '{self.prediction_column}' not found.")
        if self.target_column not in self._data.columns:
            raise ValueError(f"Label column '{self.target_column}' not found.")
        if self.metadata_columns:
            missing_columns = set(self.metadata_columns) - set(self._data.columns)
            if missing_columns:
                raise ValueError(f"Metadata columns {missing_columns} not found.")

    def load_prediction(self, index: int) -> TargetType:
        """Returns the prediction for the given index."""
        return self._data.iloc[index][self.prediction_column]

    def load_target(self, index: int) -> TargetType:
        """Returns the target for the given index."""
        return self._data.iloc[index][self.target_column]

    def load_text(self, index: int) -> MessageSeries | None:
        """Returns the text for the given index."""
        if self.text_column is None:
            return None

        text = self._data.iloc[index][self.text_column]

        try:
            return message_utils.deserialize(self._data.iloc[index][self.text_column])
        except Exception:
            return [UserMessage(content=text)]

    def load_metadata(self, index: int) -> Dict[str, Any] | None:
        """Returns the metadata for the given index."""
        if self.metadata_columns is None:
            return None

        row = self._data.iloc[index]
        return {col: row[col] for col in self.metadata_columns}

    def _apply_transforms(
        self, sample: PredictionSample[TargetType]
    ) -> PredictionSample[TargetType]:
        """Applies the dataset transforms to the prediction and target."""
        if self.transforms:
            text = self.transforms.text(sample.text) if self.transforms.text else sample.text
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
                text=text,
                metadata=sample.metadata,
            )
        return sample
