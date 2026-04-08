from typing import Any
from pydantic import BaseModel, model_validator


class Prompt(BaseModel):
    """
    Represents a single user prompt input.

    Attributes:
        prompt (str): The natural language input from the user.
    """

    prompt: str

    @model_validator(mode='before')
    @classmethod
    def check_fields(cls, values: Any) -> Any:
        """
        Validates the prompt structure before initialization.

        Args:
            values (Any): Raw input data to be validated.

        Returns:
            Any: The validated input data.

        Raises:
            ValueError: If the 'prompt' field is missing or not a string.
        """
        if not isinstance(values, dict):
            return values

        errors = []
        prompt_value = values.get('prompt')

        if prompt_value is None:
            errors.append("'prompt' field is missing.")
        elif not isinstance(prompt_value, str):
            got_type = type(prompt_value).__name__
            errors.append(f"'prompt' must be a string, got {got_type}")

        if errors:
            raise ValueError("\n    ".join(errors))

        return values
