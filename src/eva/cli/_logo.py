"""CLI logos."""

_EVA_LOGO: str = r"""
      _____   ____ _
     / _ \ \ / / _` |
    |  __/\ V / (_| |
     \___| \_/ \__,_|

         kaiko.ai
"""


ANSI_COLOR_RESET = "\33[0m"
"""ANSI color reset code."""


def _print_logo(
    logo: str,
    prefix: str = "",
    suffix: str = "",
    ansi_color: str = "\33[0;35m",
) -> None:
    r"""Prints an ASCII terminal art logo in terminal.

    Args:
        logo: The desired art logo to print.
        prefix: Characters to add before the logo. Defaults to "".
        suffix: Characters to add after the logo. Defaults to "".
        ansi_color: The color of the output. Defaults to "\33[0;32m".
    """
    colored_logo = f"{ansi_color}{logo}{ANSI_COLOR_RESET}"
    print(prefix + colored_logo + suffix)


def print_cli_logo() -> None:
    """Prints the CLI logo."""
    _print_logo(_EVA_LOGO, suffix="\n")
