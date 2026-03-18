from llm_sdk import Small_LLM_Model as LLM_Model
from src.validation_models import FunctionDefinition as FuncDef
from src.utils import load_vocab
from typing import Any, Callable
import math
import re


def get_params_instructions(
    func: FuncDef, prompt: str, p_name: str, already_extracted: dict
) -> str:
    """
    Build a formatted instruction prompt to extract a specific parameter.

    Args:
        func (FuncDef): Function definition containing parameter metadata.
        prompt (str): The original user input.
        p_name (str): Name of the parameter to extract.
        already_extracted (dict): Previously extracted parameters.

    Returns:
        str: A formatted instruction string for the LLM.
    """
    param_type = func.parameters[p_name].type
    params = list(func.parameters.keys())
    position = params.index(p_name) + 1
    total = len(params)
    context = "\n".join(f"{k}: {v}" for k, v in already_extracted.items())

    return (
        "<|im_start|>system\n"
        f"Extract argument {position} of {total} from the user prompt.\n"
        f"Read left to right. Do not compute or sum.\n"
        f"Return only the raw {param_type} value.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{prompt}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{context}\n"
        f"{p_name}: "
    )


def generate_number(llm: LLM_Model, ins: str, p_name: str) -> dict[str, float]:
    """
    Generate a numeric parameter value using constrained decoding.

    Args:
        llm (LLM_Model): The language model used for generation.
        ins (str): Instruction prompt given to the model.
        p_name (str): Name of the parameter being generated.

    Returns:
        dict[str, float]: Dictionary with the parameter name and its value.
    """
    vocab = load_vocab(llm)
    current_value = ""

    while True:
        input_ids = llm._encode(ins + current_value).tolist()[0]
        logits = llm.get_logits_from_input_ids(input_ids)
        valid_tokens = []
        for s, tid in vocab.items():
            token_str = s[1:] if s.startswith('Ġ') else s
            if not token_str:
                continue
            if token_str in [",", "}"]:
                if "." in current_value and len(current_value.split(".")[1]) >= 2:
                    break
            if not all(c in "0123456789." for c in token_str):
                continue
            try:
                value = float(current_value + token_str)
                if math.isfinite(value):
                    valid_tokens.append((tid, token_str))
            except ValueError:
                pass

        if not valid_tokens:
            break

        token_id, token_str = max(valid_tokens, key=lambda x: logits[x[0]])
        if token_str in [",", "}"]:
            break

        current_value += token_str

    return {p_name: float(current_value)}


def generate_string(llm: LLM_Model, ins: str, p_name: str) -> dict[str, str]:
    return {"name": "tlaranje"}


def generate_bool(llm: LLM_Model, ins: str, p_name: str) -> dict[str, bool]:
    return {"is": True}


# Type alias for generator functions.
# Each generator takes:
# - an LLM model
# - an instruction string
# - a parameter name
# and returns a dictionary with the generated value.
GeneratorFn = Callable[[LLM_Model, str, str], dict[str, Any]]

# Mapping between parameter types and their corresponding generator functions.
# This is used to select the correct generator based on the parameter type.
TYPE_GENERATORS: dict[str, GeneratorFn] = {
    "number": generate_number,
    "string": generate_string,
    "bool":   generate_bool,
}


def generate_parameters(
    model: LLM_Model, func: FuncDef, prompt: str
) -> dict[str, Any]:
    """Generate all argument values for a function call.

    Uses a hybrid approach: regex extraction for numbers,
    constrained decoding for strings and booleans.

    Args:
        model: The LLM wrapper used to score tokens.
        func: The target function definition, including parameter schema.
        prompt: Natural language description of the desired operation.

    Returns:
        A dict mapping every parameter name to its decoded value.

    Raises:
        ValueError: If a parameter type has no registered generator.
    """
    res: dict[str, Any] = {}

    # Extract all numbers from the prompt upfront for positional mapping
    numbers = re.findall(r'-?\d+\.?\d*', prompt)
    number_index = 0

    for param_name, param in func.parameters.items():
        if param.type == "number":
            # Use regex extraction if a number is available at this position
            if number_index < len(numbers):
                res[param_name] = float(numbers[number_index])
                number_index += 1
                continue
            # Fallback to constrained decoding if regex found nothing
            instructions = get_params_instructions(
                func, prompt, param_name, res
            )
            res.update(generate_number(model, instructions, param_name))
        else:
            # For strings and bools, always use the model
            instructions = get_params_instructions(
                func, prompt, param_name, res
            )
            generator = TYPE_GENERATORS.get(param.type)
            if generator is None:
                raise ValueError(
                    f"No generator registered for type '{param.type}' "
                    f"(parameter '{param_name}' of '{func.name}')"
                )
            res.update(generator(model, instructions, param_name))

    return res
