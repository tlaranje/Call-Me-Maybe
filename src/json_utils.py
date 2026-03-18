from typing import Any
import json


def load_json(path: str) -> Any:
    """
    Load and parse a JSON file from the given path.

    Args:
        path (str): Path to the JSON file to load.

    Returns:
        Any: Parsed JSON content as Python data structures.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains invalid JSON.
    """
    try:
        with open(path, "r") as fd:
            return json.load(fd)
    except FileNotFoundError:
        raise FileNotFoundError("File not found!")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format! - {path}")
