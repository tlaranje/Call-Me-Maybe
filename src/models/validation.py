from typing import Any
from pydantic import ValidationError

from .function_definition import FunctionDefinition
from .promt import Prompt
from src.utils import load_json


def load_and_validate(args: dict[str, str]) -> dict[str, list[Any]]:
    """
    Load and validate prompts and function definitions.

    Parses JSON files and validates them using Pydantic models,
    collecting all validation errors before raising.

    Args:
        args (dict[str, str]): Dictionary with:
            - "input": prompts file path
            - "functions": function definitions file path

    Returns:
        dict[str, list[Any]]: Validated data containing:
            - "prompts": list of Prompt objects
            - "functions": list of FunctionDefinition objects

    Raises:
        ValueError: If validation errors are found.
    """
    prompt_errors = []
    fn_errors = []

    prompts = []

    # Validate prompts
    for p in load_json(args['input']):
        try:
            prompts.append(Prompt(**p))
        except ValidationError as e:
            for error in e.errors():
                msg = error['msg'].removeprefix("Value error, ")
                prompt_errors.append(f"    {msg}")

    functions = []

    # Validate functions
    for fn in load_json(args['functions']):
        try:
            functions.append(FunctionDefinition(**fn))
        except ValidationError as e:
            for error in e.errors():
                msg = error['msg'].removeprefix("Value error, ")
                fn_errors.append(f"    {msg}")

    # Aggregate errors
    errors = []

    if prompt_errors:
        errors.append("Errors in function_calling_tests.json:")
        errors.extend(prompt_errors)

    if fn_errors:
        errors.append("Errors in functions_definition.json:")
        errors.extend(fn_errors)

    # Raise only if there are actual errors
    if errors:
        raise ValueError("\n".join(errors))

    return {"prompts": prompts, "functions": functions}
