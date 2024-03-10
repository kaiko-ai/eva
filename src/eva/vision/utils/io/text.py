"""Text I/O related functions."""

import csv
from typing import Dict, List


def read_csv(path: str) -> List[Dict[str, str]]:
    """Reads a CSV file and returns its contents as a list of dictionaries.

    Args:
        path: The path to the CSV file.

    Returns:
        A list of dictionaries representing the data in the CSV file.
    """
    with open(path, newline="") as file:
        data = csv.DictReader(file, skipinitialspace=True)
        return list(data)
