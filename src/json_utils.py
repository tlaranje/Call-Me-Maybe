from src.constants import Colors as C
from typing import Any
import json


def load_json_file(path: str) -> Any:
    try:
        with open(path, "r") as fd:
            return json.load(fd)
    except FileNotFoundError:
        raise FileNotFoundError(f"{C.RED}File not found!{C.END}")
    except json.JSONDecodeError:
        raise ValueError(f"{C.RED}Invalid JSON format in file!{C.END}")
