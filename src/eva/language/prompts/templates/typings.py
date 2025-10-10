from typing_extensions import override, NotRequired, TypedDict, List

class QuestionAnswerExample(TypedDict):
    question: str
    answer: str
    context: NotRequired[str | List[str]]
