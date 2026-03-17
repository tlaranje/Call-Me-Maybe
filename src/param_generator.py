from llm_sdk import Small_LLM_Model as LLM_Model
from src.validation_models import FunctionDefinition as FuncDef
from src.utils import load_vocab
from typing import Any, Callable
import json
import math


def get_params_instructions(func: FuncDef, prompt: str, p_name: str) -> str:
    """Build the instruction prompt for extracting a single parameter value.

    Args:
        func: The function definition containing parameter schemas.
        prompt: Natural language description of the desired operation.
        p_name: The specific parameter name to extract.

    Returns:
        A formatted instruction string for the LLM.
    """
    param_type = func.parameters[p_name].type

    return (
        "<|im_start|>system\n"
        f"Extract the value of '{p_name}' ({param_type}) "
        f"from the user prompt.\n"
        f"Reply with only the {param_type} value followed by '}}'\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{prompt}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{p_name}: "  # força o modelo a começar pelo valor diretamente
    )


def generate_number(llm: LLM_Model, ins: str, p_name: str) -> dict[str, float]:
    """Generate a numeric argument value using constrained decoding.

    Args:
        llm: The LLM wrapper used to score tokens.
        ins: The instruction prompt for this parameter.
        p_name: The parameter name to generate a value for.

    Returns:
        A dict mapping the parameter name to its decoded float value.
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
                valid_tokens.append((tid, token_str))
                continue
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


GeneratorFn = Callable[[LLM_Model, str, str], dict[str, Any]]

TYPE_GENERATORS: dict[str, GeneratorFn] = {
    "number": generate_number,
    "string": generate_string,
    "bool":   generate_bool,
}


def generate_parameters(
    model: LLM_Model, func: FuncDef, prompt: str
) -> dict[str, Any]:
    res = {}
    for param_name, param in func.parameters.items():
        instructions = get_params_instructions(func, prompt, param_name)
        generator = TYPE_GENERATORS.get(param.type)
        if generator is None:
            raise ValueError(
                f"No generator registered for type '{param.type}' "
                f"(parameter '{param_name}' of '{func.name}')"
            )

        res.update(generator(model, instructions, param_name))
    return res
