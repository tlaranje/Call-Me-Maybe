from pydantic import BaseModel


class FunctionParameter(BaseModel):
    """
    Represents a single parameter in a function definition.

    Attributes:
        type (str): The expected data type of the parameter.
    """

    type: str
