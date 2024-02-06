"""EVA's main cli manager."""

import jsonargparse

from eva import interface
from eva.cli import _logo


def cli() -> object:
    """Main CLI factory."""
    _logo.print_cli_logo()
    return jsonargparse.CLI(interface.Interface, parser_mode="omegaconf", fail_untyped=False)
