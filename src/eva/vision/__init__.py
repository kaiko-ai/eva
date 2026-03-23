"""eva vision API."""

try:
    from eva.vision import callbacks, losses, models, utils
    from eva.vision.data import datasets, transforms
except ImportError as e:
    msg = (
        "eva vision requirements are not installed.\n\n"
        "Please pip install as follows:\n"
        '  python -m pip install "kaiko-eva[vision]" --upgrade'
    )
    raise ImportError(str(e) + "\n\n" + msg) from e

__all__ = ["callbacks", "losses", "models", "utils", "datasets", "transforms"]
