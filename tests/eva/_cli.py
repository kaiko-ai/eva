"""CLI test and helper utilities."""

import sys
from types import ModuleType
from typing import List

from eva import __main__


def call_module_with_cli_arguments(
    module: ModuleType,
    cli_arguments: List[str],
) -> None:
    """Calls a module with arguments like they were passed on the command-line.

    Args:
        module: An python module with a `main()` function.
        cli_arguments: the arguments that are passed on the command-line.
    """
    sys.argv = ["test"] + cli_arguments

    try:
        module.main()

    except SystemExit as system_exception:
        if system_exception.code != 0:
            raise

    except Exception as exception:
        raise exception


def run_cli_from_main(cli_args: List[str]) -> None:
    """Runs CLI from `__main__`."""
    call_module_with_cli_arguments(__main__, cli_args)
