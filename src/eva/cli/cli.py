"""EVA's main cli manager."""

import jsonargparse

from eva import interface
from eva.cli import logo, setup


def cli() -> object:
    """Main CLI factory."""
    logo.print_cli_logo()
    setup.setup()
    return jsonargparse.CLI(
        interface.Interface,
        parser_mode="omegaconf",
        fail_untyped=False,
    )
