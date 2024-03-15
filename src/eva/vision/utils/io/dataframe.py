"""DataFrame related I/O operations."""

import pandas as pd


def read_dataframe(path: str) -> pd.DataFrame:
    """Reads and loads a DataFrame file.

    Args:
        path: The path to the manifest file.

    Returns:
        The data of the file as a `DataFrame`.
    """
    if path.endswith(".csv"):
        data = pd.read_csv(path)
    elif path.endswith(".parquet"):
        data = pd.read_parquet(path)
    else:
        raise ValueError(f"Failed to load manifest file at '{path}'.")
    return data
