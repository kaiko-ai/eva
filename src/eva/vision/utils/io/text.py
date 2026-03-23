"""Text I/O related functions."""

import csv
from typing import Dict, List


def read_csv(
    path: str,
    *,
    delimiter: str = ",",
    encoding: str = "utf-8",
) -> List[Dict[str, str]]:
    """Reads a CSV file and returns its contents as a list of dictionaries.

    Args:
        path: The path to the CSV file.
        delimiter: The character that separates fields in the CSV file.
        encoding: The encoding of the CSV file.

    Returns:
        A list of dictionaries representing the data in the CSV file.
    """
    with open(path, newline="", encoding=encoding) as file:
        data = csv.DictReader(file, skipinitialspace=True, delimiter=delimiter)
        return list(data)
