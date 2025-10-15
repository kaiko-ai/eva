"""Model postprocessing transforms."""

from eva.language.models.postprocess.extract_answer_from_json import ExtractDiscreteAnswerFromJson
from eva.language.models.postprocess.str_to_int_tensor import CastStrToIntTensor

__all__ = ["CastStrToIntTensor", "ExtractDiscreteAnswerFromJson"]
