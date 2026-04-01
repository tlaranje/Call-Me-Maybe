from typing import Any
import json


def load_json(path: str) -> Any:
    """
    Load and parse a JSON file from the given path.

    Args:
        path (str): Path to the JSON file.

    Returns:
        Any: Parsed JSON content.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains invalid JSON.
    """
    try:
        # Open and parse JSON file
        with open(path, "r") as fd:
            return json.load(fd)

    except FileNotFoundError:
        # Preserve original error context but improve message
        raise FileNotFoundError(f"File not found: {path}")

    except json.JSONDecodeError as e:
        # Include original parsing error for debugging
        raise ValueError(f"Invalid JSON format in '{path}': {e}")
