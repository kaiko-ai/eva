"""Core I/O utilities."""

from eva.core.utils.io.dataframe import read_dataframe
from eva.core.utils.io.gz import gunzip_file

__all__ = ["read_dataframe", "gunzip_file"]
