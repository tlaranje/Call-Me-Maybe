from pydantic import BaseModel, model_validator, ValidationError
from src.json_utils import load_json
from typing import Any


class FunctionParameter(BaseModel):
    """
    Represent a single parameter in a function definition.

    Args:
        type (str): Expected type of the parameter.
    """
    type: str


class FunctionDefinition(BaseModel):
    """
    Represent a function definition.

    Args:
        name (str): Function name.
        description (str): Human-readable description.
        parameters (dict[str, FunctionParameter]): Function parameters.
        returns (FunctionParameter): Return type definition.
    """
    name: str
    description: str
    parameters: dict[str, FunctionParameter]
    returns: FunctionParameter

    @model_validator(mode='before')
    @classmethod
    def check_fields(cls, values: Any) -> Any:
        """
        Perform custom validation before model parsing.

        Ensures required fields exist and have the correct structure.

        Args:
            values (Any): Raw input data.

        Returns:
            Any: Validated values.

        Raises:
            ValueError: If validation fails.
        """
        errors = []

        # Validate name
        if values.get('name') is None:
            errors.append("'name' field is missing.")
        elif not isinstance(values.get('name'), str):
            errors.append(
                "'name' must be a string, got "
                f"{type(values.get('name')).__name__}."
            )

        # Validate description
        if values.get('description') is None:
            errors.append("'description' field is missing.")
        elif not isinstance(values.get('description'), str):
            errors.append(
                "'description' must be a string, got "
                f"{type(values.get('description')).__name__}."
            )

        # Validate parameters
        if values.get('parameters') is None:
            errors.append("'parameters' field is missing.")
        else:
            for key, value in values.get('parameters').items():
                if 'type' not in value:
                    errors.append(
                        f"parameter '{key}' must have a 'type' field."
                    )
                elif not isinstance(value['type'], str):
                    errors.append(
                        f"parameter '{key}' 'type' must be a string, got "
                        f"{type(value['type']).__name__}."
                    )

        # Validate returns
        if values.get('returns') is None:
            errors.append("'returns' field is missing.")
        elif 'type' not in values.get('returns'):
            errors.append("'returns' must have a 'type' field.")

        # Raise aggregated errors
        if errors:
            raise ValueError("\n    ".join(errors))

        return values


class Prompt(BaseModel):
    """
    Represent a single user prompt.

    Args:
        prompt (str): Natural language input.
    """
    prompt: str

    @model_validator(mode='before')
    @classmethod
    def check_fields(cls, values: Any) -> Any:
        """
        Validate prompt structure.

        Args:
            values (Any): Raw input data.

        Returns:
            Any: Validated values.

        Raises:
            ValueError: If validation fails.
        """
        errors = []

        if values.get('prompt') is None:
            errors.append("'prompt' field is missing.")
        elif not isinstance(values.get('prompt'), str):
            errors.append(
                "'prompt' must be a string, got "
                f"{type(values.get('prompt')).__name__}"
            )

        if errors:
            raise ValueError("\n    ".join(errors))

        return values


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
