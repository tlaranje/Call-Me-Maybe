from pydantic import BaseModel, model_validator, ValidationError
from src.json_utils import load_json
from typing import Any


class FunctionParameter(BaseModel):
    """
    Represent a single parameter in a function definition.

    Attributes:
        type (str): The expected type of the parameter.
    """
    type: str


class FunctionDefinition(BaseModel):
    """
    Represent a function definition with its name, description,
    parameters, and return type.

    Attributes:
        name (str): The function name.
        description (str): A human-readable description of the function.
        parameters (dict[str, FunctionParameter]): Mapping of parameter names
        to their types.
        returns (FunctionParameter): The return type definition.
    """
    name: str
    description: str
    parameters: dict[str, FunctionParameter]
    returns: FunctionParameter

    @model_validator(mode='before')
    @classmethod
    def check_fields(cls, values: Any) -> Any:
        """
        Validate that all required fields exist and have correct types.

        Args:
            values (Any): Raw values before validation.

        Returns:
            Any: Validated values.

        Raises:
            ValueError: If any required field is missing or incorrectly type.
        """
        errors = []

        if values.get('name') is None:
            errors.append("'name' field is missing.")
        elif not isinstance(values.get('name'), str):
            errors.append(
                "'name' must be a string, got "
                f"{type(values.get('name')).__name__}."
            )

        if values.get('description') is None:
            errors.append("'description' field is missing.")
        elif not isinstance(values.get('description'), str):
            errors.append(
                "'description' must be a string, got "
                f"{type(values.get('description')).__name__}."
            )

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

        if values.get('returns') is None:
            errors.append("'returns' field is missing.")
        elif 'type' not in values.get('returns'):
            errors.append("'returns' must have a 'type' field.")

        if errors:
            raise ValueError("\n    ".join(errors))

        return values


class Prompt(BaseModel):
    """
    Represent a single natural-language prompt.

    Attributes:
        prompt (str): The user-provided prompt text.
    """
    prompt: str

    @model_validator(mode='before')
    @classmethod
    def check_fields(cls, values: Any) -> Any:
        """
        Validate that the prompt field exists and is a string.

        Args:
            values (Any): Raw values before validation.

        Returns:
            Any: Validated values.

        Raises:
            ValueError: If the prompt is missing or incorrectly typed.
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
    Load input JSON files and validate them using Pydantic models.

    Args:
        args (dict[str, str]): Dictionary containing paths for:
            - "input": prompts file
            - "functions": function definitions file

    Returns:
        dict[str, list]: A dictionary containing:
            - "prompts": list of validated Prompt objects
            - "functions": list of validated FunctionDefinition objects

    Raises:
        ValueError: If any validation errors are found in the input files.
    """
    prompt_errors = []
    fn_errors = []

    prompts = []
    for p in load_json(args['input']):
        try:
            prompts.append(Prompt(**p))
        except ValidationError as e:
            for error in e.errors():
                msg = error['msg'].removeprefix("Value error, ")
                prompt_errors.append(f"    {msg}")
    prompt_errors.append("")

    functions = []
    for fn in load_json(args['functions']):
        try:
            functions.append(FunctionDefinition(**fn))
        except ValidationError as e:
            for error in e.errors():
                msg = error['msg'].removeprefix("Value error, ")
                fn_errors.append(f"    {msg}")

    errors = []
    if prompt_errors:
        errors.append("Errors in function_calling_tests.json:")
        errors.extend(prompt_errors)
    if fn_errors:
        errors.append("Errors in functions_definition.json:")
        errors.extend(fn_errors)

    if len(errors) > 2:
        raise ValueError("\n".join(errors))

    return {"prompts": prompts, "functions": functions}
