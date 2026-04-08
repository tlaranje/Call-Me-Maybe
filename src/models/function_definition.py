from typing import Any
from pydantic import BaseModel, model_validator

from .function_parameter import FunctionParameter


class FunctionDefinition(BaseModel):
    """
    Represents a complete function definition for the LLM pipeline.

    Attributes:
        name (str): The unique identifier for the function.
        description (str): A detailed explanation of what the function does.
        parameters (dict[str, FunctionParameter]): A mapping of parameter names
            to their respective type definitions.
        returns (FunctionParameter): The definition of the return type.
    """
    name: str
    description: str
    parameters: dict[str, FunctionParameter]
    returns: FunctionParameter

    @model_validator(mode='before')
    @classmethod
    def check_fields(cls, values: Any) -> Any:
        """
        Validates the function structure before parsing into Pydantic models.

        Args:
            values (Any): The raw data dictionary to validate.

        Returns:
            Any: The validated values.

        Raises:
            ValueError: If required fields are missing or have incorrect types.
        """
        if not isinstance(values, dict):
            return values

        errors = []

        # Validate basic string fields
        for field in ["name", "description"]:
            val = values.get(field)
            if val is None:
                errors.append(f"'{field}' field is missing.")
            elif not isinstance(val, str):
                got = type(val).__name__
                errors.append(f"'{field}' must be a string, got {got}.")

        # Validate parameters dictionary
        params = values.get('parameters')
        if params is None:
            errors.append("'parameters' field is missing.")
        elif isinstance(params, dict):
            for key, val in params.items():
                if not isinstance(val, dict) or 'type' not in val:
                    errors.append(f"parameter '{key}' must have a 'type'.")
                elif not isinstance(val['type'], str):
                    got = type(val['type']).__name__
                    errors.append(
                        f"parameter '{key}' type must be string, got {got}."
                    )

        # Validate returns field
        ret = values.get('returns')
        if ret is None:
            errors.append("'returns' field is missing.")
        elif not isinstance(ret, dict) or 'type' not in ret:
            errors.append("'returns' must have a 'type' field.")

        if errors:
            raise ValueError("\n    ".join(errors))

        return values
