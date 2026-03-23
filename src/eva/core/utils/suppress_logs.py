"""Context manager to temporarily suppress all logging outputs."""

import logging
import sys
from types import TracebackType
from typing import Type


class SuppressLogs:
    """Context manager to suppress all logs but print exceptions if they occur."""

    def __enter__(self) -> None:
        """Temporarily increase log level to suppress all logs."""
        self._logger = logging.getLogger()
        self._previous_level = self._logger.level
        self._logger.setLevel(logging.CRITICAL + 1)

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Restores the previous logging level and print exceptions."""
        self._logger.setLevel(self._previous_level)
        if exc_value:
            print(f"Error: {exc_value}", file=sys.stderr)
        return False
