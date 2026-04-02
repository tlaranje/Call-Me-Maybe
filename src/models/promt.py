from pydantic import BaseModel, model_validator
from typing import Any


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
