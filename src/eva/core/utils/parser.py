"""Parsing related helper functions."""

from typing import Any, Dict

import jsonargparse


def parse_object(config: Dict[str, Any], expected_type: Any = Any) -> Any:
    """Parse object which is defined as dictionary."""
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("module", type=expected_type)
    configuration = parser.parse_object({"module": config})
    init_object = parser.instantiate_classes(configuration)
    obj_module = init_object.module
    if isinstance(obj_module, jsonargparse.Namespace):
        raise ValueError(
            f"Failed to parsed object '{obj_module.class_path}'. "
            "Please check that the initialized arguments are valid."
        )
    return obj_module
