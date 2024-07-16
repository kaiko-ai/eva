"""Phikon network."""

from eva.core.models.networks import wrappers, transforms


class Phikon(wrappers.HuggingFaceModel):
    """Phikon is a self-supervised learning model for histopathology trained with iBOT.
    
    For more details, see https://huggingface.co/owkin/phikon.
    """

    def __init__(self) -> None:
        super().__init__(
            model_name_or_path="owkin/phikon",
            tensor_transforms=transforms.ExtractCLSFeatures(),
        )
