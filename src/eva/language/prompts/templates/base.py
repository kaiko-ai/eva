import abc


class PromptTemplate(abc.ABC):
    @abc.abstractmethod
    def render(self, **kwargs) -> str:
        pass
