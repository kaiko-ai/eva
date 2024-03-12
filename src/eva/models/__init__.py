"""Models API."""

<<<<<<< HEAD
from eva.models.modules import DecoderModule, HeadModule, InferenceModule
from eva.models.networks import ModelFromFunction

__all__ = ["DecoderModule", "HeadModule", "ModelFromFunction", "InferenceModule"]
=======
from eva.models.modules import HeadModule, InferenceModule

__all__ = ["HeadModule", "InferenceModule"]
>>>>>>> 225-speedup-totalsegmentator2d-io
