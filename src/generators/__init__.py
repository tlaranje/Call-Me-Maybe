from .types import Boolean, String, Integer, Float
from .function_generator import FunctionGenerator
from src.utils import render_panel
from typing import TYPE_CHECKING, Any
import time

if TYPE_CHECKING:
    from llm_sdk import Small_LLM_Model as LLM_Model
    from src.models import FunctionDefinition
    from rich.Live import Live


class FunctionCaller:
    def __init__(self):
        self.function_generator = FunctionGenerator()

        self.type_registry = {
            "number": Float(),
            "integer": Integer(),
            "float": Float(),
            "string": String(),
            "boolean": Boolean(),
        }

    def generate_parameters(
        self,
        model: LLM_Model,
        func: FunctionDefinition,
        prompt: str,
        instructions: str,
        live: Live | None = None
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

        def render(current_param: str = "", current_value: str = "") -> None:
            """
            Update UI with current generation state.
            """
            if live:
                params_display = res.copy()

                if current_param:
                    params_display[current_param] = current_value or "..."

                live.update(render_panel(prompt, func.name, params_display))

        for param_name, param in func.parameters.items():
            # Append parameter prefix to instruction
            instructions += f"{param_name}="

            generator = self.type_registry.get(param.type)

            if generator is None:
                raise ValueError(
                    f"No generator registered for type '{param.type}' "
                    f"(parameter '{param_name}' of '{func.name}')"
                )

            # Initial state for this parameter
            render(param_name, "...")

            # Generate value using appropriate generator
            value = generator.generate(model, instructions)
            res[param_name] = value

            generated = str(value)
            displayed = ""
            # Typing animation
            for char in generated:
                displayed += char
                render(param_name, displayed)
                time.sleep(0.05)

            # Append generated value to instruction context
            instructions += f"{generated}\n"

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
            "according to the user's prompt, followed by a \\n character. "
            f"{str(func)}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{prompt}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
