"""Parsing related helper functions."""

from typing import Any, Dict

import jsonargparse


def parse_object(config: Dict[str, Any]) -> Any:
    """Parse object which is defined as dictionary."""
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("module", type=Any)
    configuration = parser.parse_object({"module": config})
    init_object = parser.instantiate_classes(configuration)
    return init_object.module
