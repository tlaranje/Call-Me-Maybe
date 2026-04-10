from .function_generator import FunctionGenerator
from typing import TYPE_CHECKING, Any, Protocol
from .types import Boolean, Numbers, String

if TYPE_CHECKING:
    from llm_sdk import Small_LLM_Model as LLM_Model
    from src.models import FunctionDefinition


class GeneratorProtocol(Protocol):
    """Protocol defining the interface for all type generators."""
    def generate(self, llm: "LLM_Model", ins: str) -> Any:
        ...


class FunctionCaller:
    """
    Orchestrates the extraction of parameters for a selected function.
    """

    def __init__(self) -> None:
        """
        Initializes the caller and registers generators for each type.
        """
        self.function_generator = FunctionGenerator()

        # Mapping types to their respective generator classes
        self.type_registry: dict[str, GeneratorProtocol] = {
            "number": Numbers("float"),
            "float": Numbers("float"),
            "integer": Numbers("int"),
            "string": String(),
            "boolean": Boolean(),
        }

    def generate_parameters(
        self,
        model: "LLM_Model",
        func: "FunctionDefinition",
        prompt: str
    ) -> dict[str, Any]:
        """
        Generates all parameters for a selected function using the LLM.

        Args:
            model (LLM_Model): The language model instance.
            func (FunctionDefinition): The definition of the target function.
            prompt (str): The user's input message.

        Returns:
            dict[str, Any]: A dictionary containing the generated parameters.

        Raises:
            ValueError: If a parameter type is not supported by the registry.
        """
        res: dict[str, Any] = {}
        instructions = self.get_params_instructions(func, prompt)

        for param_name, param in func.parameters.items():
            # Apply specific syntax based on parameter type
            if param.type == "string":
                instructions += f'{param_name}='
            else:
                instructions += f'{param_name}->'

            generator = self.type_registry.get(param.type)

            if generator is None:
                raise ValueError(
                    f"No generator found for type '{param.type}' "
                    f"(param: '{param_name}' in function: '{func.name}')"
                )

            # Generate value and store it
            value = generator.generate(model, instructions)
            res[param_name] = value

            # Update instructions with the new value for the next parameter
            instructions += f"{str(value)}\n"

        return res

    def get_params_instructions(
        self,
        func: "FunctionDefinition",
        prompt: str
    ) -> str:
        """
        Builds the instruction prompt for parameter extraction.

        Args:
            func (FunctionDefinition): Function schema and descriptions.
            prompt (str): The user's natural language input.

        Returns:
            str: The formatted system and user prompt.
        """
        return (
            "<|im_start|>system\n"
            "Extract parameter values VERBATIM from the user message.\n"
            f"{str(func)} <|im_end|>\n"
            "<|im_start|>user\n"
            "Format template: Hello {user}'s profile!\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "template=Hello {user}'s profile!\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{prompt}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
