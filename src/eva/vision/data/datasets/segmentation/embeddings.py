"""TotalSegmentator 2D segmentation dataset class."""

import functools
import os
from glob import glob
from typing import Any, Callable, Dict, List, Literal, Tuple

import numpy as np
import numpy.typing as npt
import torch
import tqdm
from torchvision import tv_tensors
from torchvision.datasets import utils
from typing_extensions import override

from eva.vision.data.datasets import _utils, _validators, structs
from eva.vision.data.datasets.segmentation import base
from eva.vision.utils import io

from eva.core.data.datasets import embeddings as embeddings_base


class EmbeddingsSegmentation(base.ImageSegmentation, embeddings_base.EmbeddingsDataset[torch.Tensor]):
    """Embeddings segmentation dataset."""
