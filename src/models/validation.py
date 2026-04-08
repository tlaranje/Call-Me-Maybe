from typing import Any
from pydantic import ValidationError

from src.utils import load_json
from .function_definition import FunctionDefinition
from .promt import Prompt


def load_and_validate(args: dict[str, str]) -> dict[str, list[Any]]:
    """
    Loads and validates prompts and function definitions from JSON files.

    Args:
        args (dict[str, str]): Dictionary containing "input" and "functions"
            file paths.

    Returns:
        dict[str, list[Any]]: Validated data with "prompts" and "functions".

    Raises:
        ValueError: If any validation errors are found in the input files.
    """
    prompt_errors = []
    fn_errors = []
    prompts = []
    functions = []

    # Validate prompts
    for p in load_json(args['input']):
        try:
            prompts.append(Prompt(**p))
        except ValidationError as e:
            for error in e.errors():
                msg = error['msg'].replace("Value error, ", "")
                prompt_errors.append(f"    {msg}")

    # Validate functions
    for fn in load_json(args['functions']):
        try:
            functions.append(FunctionDefinition(**fn))
        except ValidationError as e:
            for error in e.errors():
                msg = error['msg'].replace("Value error, ", "")
                fn_errors.append(f"    {msg}")

    # Aggregate errors
    errors = []
    if prompt_errors:
        errors.append("Errors in function_calling_tests.json:")
        errors.extend(prompt_errors)

    if fn_errors:
        errors.append("Errors in functions_definition.json:")
        errors.extend(fn_errors)

    if errors:
        raise ValueError("\n".join(errors))

    return {"prompts": prompts, "functions": functions}
