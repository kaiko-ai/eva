"""Model outputs transforms API."""

from eva.core.models.transforms.as_discrete import AsDiscrete
from eva.core.models.transforms.extract_cls_features import ExtractCLSFeatures
from eva.core.models.transforms.extract_patch_features import ExtractPatchFeatures

__all__ = ["AsDiscrete", "ExtractCLSFeatures", "ExtractPatchFeatures"]
