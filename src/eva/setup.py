"""Operations which are executed with the package import."""

import os
import sys
import warnings


def _suppress_warnings() -> None:
    """Suppress all warnings from all subprocesses."""
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"


def _enable_mps_fallback() -> None:
    """If not set, it enables the MPS fallback in torch.

    Note that this action has to take place before importing torch.
    """
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") is None:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def setup() -> None:
    """Sets up the environment before the module is imported."""
    _suppress_warnings()
    _enable_mps_fallback()


setup()
