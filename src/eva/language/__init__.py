"""eva language API."""

try:
    from eva.language.data import datasets
except ImportError as e:
    msg = (
        "eva language requirements are not installed.\n\n"
        "Please pip install as follows:\n"
        '  python -m pip install "kaiko-eva[language]" --upgrade'
    )
    raise ImportError(str(e) + "\n\n" + msg) from e

__all__ = ["datasets"]
