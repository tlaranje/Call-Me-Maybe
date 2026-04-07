from .types import Boolean, String, Numbers
from .function_generator import FunctionGenerator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llm_sdk import Small_LLM_Model as LLM_Model
    from src.models import FunctionDefinition


class FunctionCaller:
    def __init__(self):
        self.function_generator = FunctionGenerator()

        self.type_registry = {
            "number": Numbers("float"),
            "float": Numbers("float"),
            "integer": Numbers("int"),
            "string": String(),
            "boolean": Boolean(),
        }

    def generate_parameters(
        self,
        model: LLM_Model,
        func: FunctionDefinition,
        prompt: str
    ) -> dict[str, Any]:
        """
        Generate all parameters for a selected function.

        Iterates through each parameter definition and generates values
        using type-specific generators, with optional live UI updates.

        Args:
            model (LLM_Model): Language model used for generation.
            func (FuncDef): Function definition.
            prompt (str): User input prompt.
            instructions (str): Base instruction string.
            live (optional): Rich Live instance for UI updates.

        Returns:
            dict[str, Any]: Dictionary with generated parameters.
        """
        res: dict[str, Any] = {}

        instructions = self.get_params_instructions(func, prompt)
        for param_name, param in func.parameters.items():
            # Append parameter prefix to instruction
            instructions += f"{param_name}="

            generator = self.type_registry.get(param.type)

            if generator is None:
                raise ValueError(
                    f"No generator registered for type '{param.type}' "
                    f"(parameter '{param_name}' of '{func.name}')"
                )

            # Generate value using appropriate generator
            value = generator.generate(model, instructions)
            res[param_name] = value

            # Append generated value to instruction context
            instructions += f"{str(value)}\n"

        return res

    def get_params_instructions(
        self, func: FunctionDefinition, prompt: str
    ) -> str:
        """
        Build the base instruction prompt for parameter extraction.

        Args:
            func (FuncDef): Function definition containing parameter schema.
            prompt (str): User's natural language input.

        Returns:
            str: Formatted instruction string for the LLM.
        """
        return (
            "<|im_start|>system\n"
            "Select the arguments for the following function "
            "according to the user's prompt.\n"
            f"{str(func)} <|im_end|>\n"
            "<|im_start|>user\n"
            f"{prompt}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
