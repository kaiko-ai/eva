"""Processing workers utilities and helper functions."""

import multiprocessing
from typing import Any, Callable


def main_worker_only(func: Callable) -> Any:
    """Function decorator which will execute it only on main / worker process."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Wrapper function for the decorated method."""
        if is_main_worker():
            return func(*args, **kwargs)

    return wrapper


def is_main_worker() -> bool:
    """Returns whether the main process / worker is currently used."""
    process = multiprocessing.current_process()
    return process.name == "MainProcess"
