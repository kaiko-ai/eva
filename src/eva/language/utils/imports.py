"""Utility functions for imports."""


def is_vllm_available() -> bool:
    """Check if vLLM is available.

    Returns:
        True if vLLM is installed, False otherwise.
    """
    try:
        import vllm  # type: ignore[import-not-found]  # noqa: F401

        from eva.language.models.wrappers import VllmModel  # noqa: F401

        return True
    except ImportError:
        return False
