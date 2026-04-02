from pydantic import BaseModel, model_validator
from typing import Any
from .function_parameter import FunctionParameter


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
