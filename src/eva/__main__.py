"""Main entry-point module."""

from eva.core.cli import cli


def main() -> None:
    """Main entry-point.

    The CLI fetches the input arguments from `sys.argv`.

    For usage information, execute:
        $ eva --help
    """
    cli()


if __name__ == "__main__":
    main()
