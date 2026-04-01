from src.validation_models import FunctionDefinition as FuncDef
from llm_sdk import Small_LLM_Model as LLM_Model
from src.utils import render_panel, load_vocab
from typing import Any, Callable
from rich.live import Live
import math
import time


def get_params_instructions(func: FuncDef, prompt: str) -> str:
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


def generate_number(llm: LLM_Model, ins: str, p_name: str) -> dict[str, float]:
    """
    Generate a numeric parameter using constrained decoding.

    Builds a number token-by-token, only accepting tokens that
    keep the string parsable as a valid finite float.

    Args:
        llm (LLM_Model): Language model used for generation.
        ins (str): Instruction prompt.
        p_name (str): Parameter name.

    Returns:
        dict[str, float]: Generated numeric value mapped to parameter name.
    """
    curr_value = ""
    curr_token = ""

    # Initial encoding
    input_ids = llm.encode(ins + curr_value).tolist()[0]
    logits = llm.get_logits_from_input_ids(input_ids)
    vocab = load_vocab(llm)

    while curr_token != 'Ċ':
        # Select highest probability token (greedy decoding)
        curr_token = max(vocab.keys(), key=lambda s: logits[vocab[s]])

        try:
            # Check if token keeps valid float
            value = float(curr_value + curr_token)

            if math.isfinite(value):
                # Accept token and update context
                curr_value += curr_token
                input_ids = llm.encode(ins + curr_value).tolist()[0]
                logits = llm.get_logits_from_input_ids(input_ids)
            else:
                # Reject non-finite numbers
                logits[vocab[curr_token]] = -math.inf

        except ValueError:
            # Reject invalid float sequences
            logits[vocab[curr_token]] = -math.inf

    return {p_name: float(curr_value)}


def generate_string(llm: LLM_Model, ins: str, p_name: str) -> dict[str, str]:
    """
    Generate a string parameter using constrained decoding.

    Handles BPE tokens (e.g., 'Ġ' for spaces) and reconstructs
    a natural string until a newline token is reached.

    Args:
        llm (LLM_Model): Language model used for generation.
        ins (str): Instruction prompt.
        p_name (str): Parameter name.

    Returns:
        dict[str, str]: Generated string mapped to parameter name.
    """
    curr_value = ""
    curr_token = ""

    input_ids = llm.encode(ins + curr_value).tolist()[0]
    logits = llm.get_logits_from_input_ids(input_ids)
    vocab = load_vocab(llm)

    while curr_token != 'Ċ':
        # Greedy token selection
        raw_token = max(vocab.keys(), key=lambda s: logits[vocab[s]])

        # Handle BPE space prefix
        if raw_token.startswith('Ġ') and curr_value:
            curr_token = ' ' + raw_token[1:]
        else:
            curr_token = (
                raw_token[1:] if raw_token.startswith('Ġ') else raw_token
            )

        # Skip empty tokens
        if not curr_token:
            logits[vocab[raw_token]] = -math.inf
            continue

        # Stop condition (newline token)
        if 'Ċ' in curr_token:
            curr_value += curr_token.split('Ċ')[0]
            break

        # Append token and continue decoding
        curr_value += curr_token
        input_ids = llm.encode(ins + curr_value).tolist()[0]
        logits = llm.get_logits_from_input_ids(input_ids)

        # Fix escaped characters
        curr_value = curr_value.replace('\\\\', '\\')

    return {p_name: curr_value}


# Type alias for generator functions
GeneratorFn = Callable[[LLM_Model, str, str], dict[str, Any]]

# Mapping between parameter types and generators
TYPE_GENERATORS: dict[str, GeneratorFn] = {
    "number": generate_number,
    "string": generate_string
}


def generate_parameters(
    model: LLM_Model,
    func: FuncDef,
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

        generator = TYPE_GENERATORS.get(param.type)

        if generator is None:
            raise ValueError(
                f"No generator registered for type '{param.type}' "
                f"(parameter '{param_name}' of '{func.name}')"
            )

        # Initial state for this parameter
        render(param_name, "...")

        # Generate value using appropriate generator
        value = generator(model, instructions, param_name)
        res.update(value)

        generated = str(list(value.values())[0])
        displayed = ""
        # Typing animation
        for char in generated:
            displayed += char
            render(param_name, displayed)
            time.sleep(0.05)

        # Append generated value to instruction context
        instructions += f"{generated}\n"

    return res
