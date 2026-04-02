from pydantic import BaseModel


class FunctionParameter(BaseModel):
    """
    Represent a single parameter in a function definition.

    Args:
        type (str): Expected type of the parameter.
    """
    type: str
