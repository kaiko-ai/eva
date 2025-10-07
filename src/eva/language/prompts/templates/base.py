"""Abstract base classes for prompt templates."""

import abc


class PromptTemplate(abc.ABC):
    """Abstract base class for prompt templates."""

    @abc.abstractmethod
    def render(self, **kwargs) -> str:
        """Renders the prompt template with the given keyword arguments."""
        pass
