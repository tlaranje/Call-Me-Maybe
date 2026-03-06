from typing import Any
import json


def load_json_file(path: str) -> Any:
    try:
        with open(path, "r") as fd:
            return json.load(fd)
    except FileNotFoundError:
        raise FileNotFoundError("File not found!")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format! - {path}")
