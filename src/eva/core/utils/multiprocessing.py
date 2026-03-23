"""Multiprocessing utilities."""

import multiprocessing
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar

from eva.core.utils.progress_bar import tqdm


class Process(multiprocessing.Process):
    """Multiprocessing wrapper with logic to propagate exceptions to the parent process.

    Source: https://stackoverflow.com/a/33599967/4992248
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the process."""
        multiprocessing.Process.__init__(self, *args, **kwargs)

        self._parent_conn, self._child_conn = multiprocessing.Pipe()
        self._exception = None

    def run(self) -> None:
        """Run the process."""
        try:
            multiprocessing.Process.run(self)
            self._child_conn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._child_conn.send((e, tb))

    @property
    def exception(self):
        """Property that contains exception information from the process."""
        if self._parent_conn.poll():
            self._exception = self._parent_conn.recv()
        return self._exception

    def check_exceptions(self) -> None:
        """Check for exception propagate it to the parent process."""
        if not self.is_alive():
            if self.exception:
                error, traceback = self.exception
                sys.stderr.write(traceback + "\n")
                raise error


R = TypeVar("R")


def run_with_threads(
    func: Callable[..., R],
    items: Iterable[Tuple[Any, ...]],
    kwargs: Dict[str, Any] | None = None,
    num_workers: int = 8,
    progress_desc: Optional[str] = None,
    show_progress: bool = True,
    return_results: bool = True,
) -> List[R] | None:
    """Process items with multiple threads using ThreadPoolExecutor.

    Args:
        func: Function to execute for each item
        items: Iterable of items to process. Each item should be a tuple of
            arguments to pass to func.
        kwargs: Additional keyword arguments to pass to func.
        num_workers: Number of worker threads
        progress_desc: Description for progress bar
        show_progress: Whether to show progress bar
        return_results: Whether to return the results. If False, the function
            will return None.

    Returns:
        List of results if return_results is True, otherwise None
    """
    results: List[Any] = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(func, *args, **(kwargs or {})) for args in items]
        pbar = tqdm(total=len(futures), desc=progress_desc, disable=not show_progress, leave=False)
        for future in as_completed(futures):
            if return_results:
                results.append(future.result())
            pbar.update(1)
        pbar.close()

    return results if return_results else None
