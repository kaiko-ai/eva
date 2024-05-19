"""Loggers API."""

from eva.core.loggers.dummy import DummyLogger
from eva.core.loggers.json_logger import JSONLogger

__all__ = ["DummyLogger", "JSONLogger"]
