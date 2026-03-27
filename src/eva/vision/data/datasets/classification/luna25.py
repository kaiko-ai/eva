"""LUng Nodule Analysis 2025 (LUNA25) dataset."""

import glob
import os
from typing import Callable, Literal, Sequence

import numpy as np
import numpy.linalg as npl
import pandas as pd
import scipy.ndimage as ndi
import torch
from sklearn import model_selection
from typing_extensions import override

from eva.vision.data.datasets.vision import VisionDataset
from eva.vision.data.tv_tensors import Volume


class LUNA25(VisionDataset[Volume, torch.Tensor]):
    """LUng Nodule Analysis 2025 (LUNA25) dataset.

    To download the dataset, please visit:
    https://luna25.grand-challenge.org/
    """

    patch_size: list[int] = [64, 128, 128]

    size_px: int = 64
    """Size of the patch in pixels."""

    size_mm: int = 50
    """Size of the patch in mm."""

    _val_size: float = 0.2
    """The validation size percentage of the dataset."""

    _test_size: float = 0.2
    """The test size percentage of the dataset."""

    roi_crop_size: int | Sequence[int] = (64, 64, 64)
    """Crop roi size for patches."""

    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"] | None = None,
        split_version: None = None,
        rotations: tuple | None = None,
        translations: bool | None = None,
        transform: Callable | None = None,
    ) -> None:
        """Initializes the dataset.

        Args:
            root: Path to the dataset root directory.
            split: Dataset split to use ('train', 'val', 'test' or None).
                If None, it uses the full dataset.
            split_version: Whether to use a standardized split version.
                If not, it will perform the split online.
            rotations: Tuple with the rotation ranges. If `None` it will
                automatically activated if split=="train".
            translations: Whether to apply random translations. If `None` it will
                automatically activated if split=="train".
            download: Whether to download the dataset.
            transform: A callable object for applying data transformations.
                If None, no transformations are applied.
        """
        super().__init__()

        self._root = root
        self._split = split
        self._split_version = split_version
        self._rotations = (
            ((-20, 20), (-20, 20), (-20, 20))
            if rotations is None and self._split == "train"
            else rotations
        )
        self._translations = self._split == "train" if translations is None else translations
        self._transform = transform

        self._candidates: pd.DataFrame
        self._samples_blocks: dict[str, str]
        self._samples_metadata: dict[str, str]
        self._indices: list[int]

    @override
    def configure(self) -> None:
        self._candidates = self._load_candidates()
        self._samples_blocks = self._fetch_nodule_blocks()
        self._samples_metadata = self._fetch_nodule_metadata()
        self._indices = list(range(len(self._candidates)))

    @override
    def validate(self) -> None:
        def _valid_sample(index: int) -> bool:
            sample = self._candidates.loc[self._indices[index]]
            return os.path.isfile(self._samples_blocks[sample["AnnotationID"]]) and os.path.isfile(
                self._samples_metadata[sample["AnnotationID"]]
            )

        invalid_samples = [i for i in range(len(self)) if not _valid_sample(i)]
        if invalid_samples:
            raise OSError(
                f"LUNA25 dataset contains missing/corrupted samples: {invalid_samples[:10]}"
            )

    @override
    def load_data(self, index: int) -> Volume:
        sample = self._candidates.loc[self._indices[index]]
        ct_scan = np.load(self._samples_blocks[sample["AnnotationID"]], mmap_mode="r")
        metadata = np.load(self._samples_metadata[sample["AnnotationID"]], allow_pickle=True).item()
        ct_patch = _extract_patch(
            ct_data=ct_scan,
            coord=tuple(np.array(self.patch_size) // 2),
            src_voxel_origin=metadata["origin"],
            src_world_matrix=metadata["transform"],
            src_voxel_spacing=metadata["spacing"],
            output_shape=(self.size_px, self.size_px, self.size_px),
            voxel_spacing=(
                self.size_mm / self.size_px,
                self.size_mm / self.size_px,
                self.size_mm / self.size_px,
            ),
            rotations=self._rotations,
            translations=2.5 if self._translations else None,  # radius := 2.5
            coord_space_world=False,
            mode="3D",
        )
        normalized_patch = _clip_and_scale(ct_patch.astype(np.float32))
        normalized_patch_tensor = torch.from_numpy(normalized_patch)
        return Volume(normalized_patch_tensor)

    @override
    def load_target(self, index: int) -> torch.Tensor:
        sample = self._candidates.loc[self._indices[index]]
        return torch.tensor(int(sample["label"]), dtype=torch.long)

    @override
    def __len__(self) -> int:
        return len(self._indices)

    def _load_candidates(self) -> pd.DataFrame:
        path = os.path.join(self._root, "LUNA25_Public_Training_Development_Data.csv")
        return pd.read_csv(path)

    def _fetch_nodule_blocks(self) -> dict[str, str]:
        return {
            os.path.basename(f).removesuffix(".npy"): f
            for f in glob.glob(os.path.join(self._root, "luna25_nodule_blocks", "image", "*.npy"))
        }

    def _fetch_nodule_metadata(self) -> dict[str, str]:
        return {
            os.path.basename(f).removesuffix(".npy"): f
            for f in glob.glob(
                os.path.join(self._root, "luna25_nodule_blocks", "metadata", "*.npy")
            )
        }

    def _get_split_indices(self) -> list[int]:
        train_df, val_df, test_df = _stratified_split(
            self._candidates,
            val_size=self._val_size,
            test_size=self._test_size,
        )

        subset_indices_per_split = {
            "train": self._candidates.index[self._candidates.index.isin(train_df.index)],
            "val": self._candidates.index[self._candidates.index.isin(val_df.index)],
            "test": self._candidates.index[self._candidates.index.isin(test_df.index)],
            None: self._candidates.index.tolist(),
        }
        subset_indices = subset_indices_per_split.get(self._split)
        if subset_indices is None:
            raise ValueError("Invalid data split. Use 'train', 'val', 'test' or `None`.")
        return subset_indices


def _stratified_split(
    dataframe: pd.DataFrame, val_size: float = 0.2, test_size: float = 0.15, random_state=42
):
    """Perform stratified split of the dataset into train, validation, and test sets.

    The split is based on the target label, gender, age group, and patient ID to
    prevent data leakage. This ensures balanced class distribution, demographic
    fairness, and prevents overfitting by grouping age. The competitive test set
    (15% of data by default) helps benchmark model performance with consistent
    stratification across splits.

    Stratification Process:
    1. Label: Ensure balanced distribution of the target variable
       (binary classification).
    2. Gender: Maintain demographic fairness by considering gender
       in the split.
    3. Age Group: Create an AgeGroup column to avoid overfitting to
       specific ages and maintain generalization.
    4. Patient ID: Split by patient to avoid data leakage (same
       patient should not appear in both train and validation sets).

    Strategy for Validation Set:
    - Stratify data based on label, gender, and age group.
    - Ensure that the validation set includes challenging cases, such as rare age groups
      or underrepresented genders, to better evaluate model performance.

    Initial Observations:
    - Label: Binary (0 or 1), class balance needs to be maintained in both train and
             validation sets.
    - Age: Impacts model generalization and should be considered for stratification.
    - Gender: To ensure fairness and mitigate bias in the model.
    - Study Date: Can help identify dataset shifts over time.
    - Spatial Features: CoordX, CoordY, CoordZ may be relevant for the spatial distribution
      of nodules.
    - Patient ID: Split by patient, not row, to prevent data leakage.

    Holdout Competitive Test Set:
    - Size: 15% of the total dataset (can be adjusted via test_size).
    - Stratification: Ensures stratification by label, age group, and gender group.
    - Consistency: Provides a consistent benchmark across different training and validation splits.

    Args:
        dataframe: The input dataset containing features like label, gender, age, and patient ID.
        val_size: The proportion of the data to use for validation.
        test_size: The proportion of the data to use for testing.
        random_state: The random seed used for reproducibility.

    Returns:
        A tuple with the stratified training, validation and test set.
    """
    df = dataframe.copy()
    df["AgeGroup"] = pd.qcut(df["Age_at_StudyDate"], q=5, labels=False)
    df["GenderGroup"] = df["Gender"].astype("category").cat.codes
    df["StratifyGroup"] = (
        df["label"].astype(str)
        + "_"
        + df["AgeGroup"].astype(str)
        + "_"
        + df["GenderGroup"].astype(str)
    )

    test_idx, remaining_idx = next(
        model_selection.StratifiedGroupKFold(
            n_splits=int(1 / test_size), shuffle=True, random_state=random_state
        ).split(df, df["StratifyGroup"], groups=df["PatientID"])
    )
    test_df, remaining_df = df.iloc[test_idx], df.iloc[remaining_idx]

    train_idx, val_idx = next(
        model_selection.StratifiedGroupKFold(
            n_splits=int(1 / val_size), shuffle=True, random_state=random_state
        ).split(remaining_df, remaining_df["StratifyGroup"], groups=remaining_df["PatientID"])
    )
    train_df, val_df = remaining_df.iloc[train_idx], remaining_df.iloc[val_idx]

    return train_df, val_df, test_df


# https://github.com/DIAGNijmegen/luna25-baseline-public/blob/2c53b964ddea1cc1d92fc68584b0205f905353dc/dataloader.py#L315
def _extract_patch(
    ct_data,
    coord,
    src_voxel_origin,
    src_world_matrix,
    src_voxel_spacing,
    output_shape=(64, 64, 64),
    voxel_spacing=(50.0 / 64, 50.0 / 64, 50.0 / 64),
    rotations=None,
    translations=None,
    coord_space_world=False,
    mode="2D",
):
    transform_matrix = np.eye(3)

    if rotations is not None:
        (zmin, zmax), (ymin, ymax), (xmin, xmax) = rotations

        # add random rotation
        angle_x = np.multiply(np.pi / 180.0, np.random.randint(xmin, xmax, 1))[0]
        angle_y = np.multiply(np.pi / 180.0, np.random.randint(ymin, ymax, 1))[0]
        angle_z = np.multiply(np.pi / 180.0, np.random.randint(zmin, zmax, 1))[0]

        transform_matrix_aug = np.eye(3)
        transform_matrix_aug = np.dot(
            transform_matrix_aug, _rotate_matrix_x(np.cos(angle_x), np.sin(angle_x))
        )
        transform_matrix_aug = np.dot(
            transform_matrix_aug, _rotate_matrix_y(np.cos(angle_y), np.sin(angle_y))
        )
        transform_matrix_aug = np.dot(
            transform_matrix_aug, _rotate_matrix_z(np.cos(angle_z), np.sin(angle_z))
        )

        transform_matrix = np.dot(transform_matrix, transform_matrix_aug)

    if translations is not None:
        # add random translation
        radius = np.random.random_sample() * translations
        offset = _sample_random_coordinate_on_sphere(radius=radius)
        offset = offset * (1.0 / src_voxel_spacing)

        coord = np.array(coord) + offset

    # Normalize transform matrix
    this_transform_matrix = transform_matrix

    this_transform_matrix = (
        this_transform_matrix.T
        / np.sqrt(np.sum(this_transform_matrix * this_transform_matrix, axis=1))
    ).T

    inv_src_matrix = np.linalg.inv(src_world_matrix)

    # world coord sampling
    if coord_space_world:
        override_coord = inv_src_matrix.dot(coord - src_voxel_origin)
    else:
        # image coord sampling
        override_coord = coord * src_voxel_spacing
    override_matrix = (inv_src_matrix.dot(this_transform_matrix.T) * src_voxel_spacing).T

    patch = _volume_transform(
        ct_data,
        src_voxel_spacing,
        override_matrix,
        center=override_coord,
        output_shape=np.array(output_shape),
        output_voxel_spacing=np.array(voxel_spacing),
        order=1,
        prefilter=False,
    )

    if mode == "2D":
        # replicate the channel dimension
        patch = np.repeat(patch, 3, axis=0)

    else:
        patch = np.expand_dims(patch, axis=0)

    return patch


# https://github.com/DIAGNijmegen/luna25-baseline-public/blob/2c53b964ddea1cc1d92fc68584b0205f905353dc/dataloader.py#L29
def _volume_transform(  # noqa: C901
    image,
    voxel_spacing,
    transform_matrix,
    center=None,
    output_shape=None,
    output_voxel_spacing=None,
    **argv,
):
    if "offset" in argv:
        raise ValueError(
            "Cannot supply 'offset' to scipy.ndimage.affine_transform "
            "- already used by this function"
        )
    if "output_shape" in argv:
        raise ValueError(
            "Cannot supply 'output_shape' to scipy.ndimage.affine_transform "
            "- already used by this function"
        )

    if image.ndim != len(voxel_spacing):
        raise ValueError("Voxel spacing must have the same dimensions")

    if center is None:
        voxel_center = (np.array(image.shape) - 1) / 2.0
    else:
        if len(center) != image.ndim:
            raise ValueError("center point has not the same dimensionality as the image")

        # Transform center to voxel coordinates
        voxel_center = np.asarray(center) / voxel_spacing

    transform_matrix = np.asarray(transform_matrix)
    if output_voxel_spacing is None:
        if output_shape is None:
            output_voxel_spacing = np.ones(transform_matrix.shape[0])
        else:
            output_voxel_spacing = np.ones(len(output_shape))
    else:
        output_voxel_spacing = np.array(output_voxel_spacing)

    if transform_matrix.shape[1] != image.ndim:
        raise ValueError(
            "transform_matrix does not have the correct number of columns "
            "(does not match image dimensionality)"
        )
    if transform_matrix.shape[0] != image.ndim:
        raise ValueError(
            "Only allowing square transform matrices here, even though this is unneccessary. "
            "However, one will need an algorithm here to create full rank-square matrices. "
            "'QR decomposition with Column Pivoting' would probably be a solution, but the author "
            "currently does not know what exactly this is, nor how to do this..."
        )

    # Normalize the transform matrix
    transform_matrix = np.array(transform_matrix)
    transform_matrix = (
        transform_matrix.T / np.sqrt(np.sum(transform_matrix * transform_matrix, axis=1))
    ).T
    transform_matrix = np.linalg.inv(
        transform_matrix.T
    )  # Important normalization for shearing matrices!!

    # The forwardMatrix transforms coordinates from input image space into result image space
    forward_matrix = np.dot(
        np.dot(np.diag(1.0 / output_voxel_spacing), transform_matrix),
        np.diag(voxel_spacing),
    )

    if output_shape is None:
        # No output dimensions are specified
        # Therefore we calculate the region that will span the whole image
        # considering the transform matrix and voxel spacing.
        image_axes = [[0 - o, x - 1 - o] for o, x in zip(voxel_center, image.shape, strict=False)]
        image_corners = _calculate_all_permutations(image_axes)

        transformed_image_corners = (np.dot(forward_matrix, x) for x in image_corners)
        output_shape = [
            1 + int(np.ceil(2 * max(abs(x_max), abs(x_min))))
            for x_min, x_max in zip(
                np.amin(transformed_image_corners, axis=0),  # type: ignore
                np.amax(transformed_image_corners, axis=0),  # type: ignore
                strict=False,  # type: ignore
            )
        ]
    else:
        # Check output_shape
        if len(output_shape) != transform_matrix.shape[1]:
            raise ValueError("output dimensions must match dimensionality of the transform matrix")
    output_shape = np.array(output_shape)

    # Calculate the backwards matrix which will be used for the slice extraction
    backwards_matrix = npl.inv(forward_matrix)
    target_image_offset = voxel_center - backwards_matrix.dot((output_shape - 1) / 2.0)

    return ndi.affine_transform(
        image,
        backwards_matrix,
        offset=target_image_offset,
        output_shape=output_shape,
        **argv,
    )


# https://github.com/DIAGNijmegen/luna25-baseline-public/blob/2c53b964ddea1cc1d92fc68584b0205f905353dc/dataloader.py#L177
def _clip_and_scale(npzarray, max_hu=400.0, min_hu=-1000.0):
    npzarray = (npzarray - min_hu) / (max_hu - min_hu)
    npzarray[npzarray > 1] = 1.0
    npzarray[npzarray < 0] = 0.0
    return npzarray


# https://github.com/DIAGNijmegen/luna25-baseline-public/blob/2c53b964ddea1cc1d92fc68584b0205f905353dc/dataloader.py#L184
def _rotate_matrix_x(cos_angle, sin_angle):
    return np.asarray([[1, 0, 0], [0, cos_angle, -sin_angle], [0, sin_angle, cos_angle]])


# https://github.com/DIAGNijmegen/luna25-baseline-public/blob/2c53b964ddea1cc1d92fc68584b0205f905353dc/dataloader.py#L188
def _rotate_matrix_y(cos_angle, sin_angle):
    return np.asarray([[cos_angle, 0, sin_angle], [0, 1, 0], [-sin_angle, 0, cos_angle]])


# https://github.com/DIAGNijmegen/luna25-baseline-public/blob/2c53b964ddea1cc1d92fc68584b0205f905353dc/dataloader.py#L192
def _rotate_matrix_z(cos_angle, sin_angle):
    return np.asarray([[cos_angle, -sin_angle, 0], [sin_angle, cos_angle, 0], [0, 0, 1]])


# https://github.com/DIAGNijmegen/luna25-baseline-public/blob/2c53b964ddea1cc1d92fc68584b0205f905353dc/dataloader.py#L304
def _sample_random_coordinate_on_sphere(radius):
    # Generate three random numbers x,y,z using Gaussian distribution
    random_nums = np.random.normal(size=(3,))

    # You should handle what happens if x=y=z=0.
    if np.all(random_nums == 0):
        return np.zeros((3,))

    # Normalise numbers and multiply number by radius of sphere
    return random_nums / np.sqrt(np.sum(random_nums * random_nums)) * radius


# https://github.com/DIAGNijmegen/luna25-baseline-public/blob/2c53b964ddea1cc1d92fc68584b0205f905353dc/dataloader.py#L12
def _calculate_all_permutations(item_list):
    if len(item_list) == 1:
        return [[i] for i in item_list[0]]
    sub_permutations = _calculate_all_permutations(item_list[1:])
    return [[i] + p for i in item_list[0] for p in sub_permutations]
