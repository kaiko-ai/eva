"""EVA's main entry-point module."""
from eva.cli import cli


def main() -> None:
    """EVA's main entry-point.

    The CLI fetches the input arguments from `sys.argv`.

    For usage information, execute:
        $ eva --help
    """
    cli()


if __name__ == "__main__":
    main()
