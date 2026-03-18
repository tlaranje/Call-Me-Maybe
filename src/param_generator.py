from llm_sdk import Small_LLM_Model as LLM_Model
from src.validation_models import FunctionDefinition as FuncDef
from src.utils import load_vocab
from typing import Any, Callable
import math


def get_params_instructions(func: FuncDef, prompt: str) -> str:
    """Build the base instruction prompt for parameter extraction.

    Args:
        func: The function definition.
        prompt: The user's natural language prompt.

    Returns:
        A formatted base instruction string for the LLM.
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
    Generate a numeric value using constrained decoding. The function builds
    the number token by token, only accepting tokens that keep the string as
    a valid float. Generation stops when the special token 'Ċ' appears.

    Args:
        llm (LLM_Model): The language model used to generate tokens.
        ins (str): Instruction prompt given to the model.
        p_name (str): Name of the parameter to return.

    Return:
        dict[str, float]: A dictionary mapping the parameter name to the
        generated numeric value.
    """
    curr_value = ""
    curr_token = ""

    # Encode the initial prompt
    input_ids = llm._encode(ins + curr_value).tolist()[0]

    # Get initial logits and vocabulary
    logits = llm.get_logits_from_input_ids(input_ids)
    vocab = load_vocab(llm)

    # Keep generating until the stop token appears
    while curr_token != 'Ċ':
        # Pick the token with the highest logit
        curr_token = max(vocab.keys(), key=lambda s: logits[vocab[s]])

        try:
            # Check if adding the token still forms a valid float
            value = float(curr_value + curr_token)

            if math.isfinite(value):
                # Accept the token and update context
                curr_value += curr_token
                input_ids = llm._encode(ins + curr_value).tolist()[0]
                logits = llm.get_logits_from_input_ids(input_ids)
            else:
                # Reject tokens that produce non‑finite numbers
                logits[vocab[curr_token]] = -math.inf

        except ValueError:
            # Reject tokens that break float parsing
            logits[vocab[curr_token]] = -math.inf

    return {p_name: float(curr_value)}


def generate_string(llm: LLM_Model, ins: str, p_name: str) -> dict[str, str]:
    """
    Generate a string value using constrained decoding. The function builds
    the string token by token, stopping when the newline token 'Ċ' appears.
    Leading space markers ('Ġ') are converted to real spaces, preserving
    the original whitespace from the prompt.

    Args:
        llm (LLM_Model): The language model used to generate tokens.
        ins (str): Instruction prompt given to the model.
        p_name (str): Name of the parameter to return.

    Return:
        dict[str, str]: A dictionary mapping the parameter name to the
        generated string value.
    """
    curr_value = ""
    curr_token = ""

    input_ids = llm._encode(ins + curr_value).tolist()[0]
    logits = llm.get_logits_from_input_ids(input_ids)
    vocab = load_vocab(llm)

    while curr_token != 'Ċ':
        # Greedy decoding — pick the token with the highest probability
        raw_token = max(vocab.keys(), key=lambda s: logits[vocab[s]])

        # BPE tokenizers use Ġ to indicate a space before the token
        if raw_token.startswith('Ġ') and curr_value:
            curr_token = ' ' + raw_token[1:]
        else:
            curr_token = (
                raw_token[1:] if raw_token.startswith('Ġ') else raw_token
            )

        # Pure whitespace token — set its logit to -inf and retry
        if not curr_token:
            logits[vocab[raw_token]] = -math.inf
            continue

        # End-of-line token found — flush remaining chars and break
        if 'Ċ' in curr_token:
            curr_value += curr_token.split('Ċ')[0]
            break

        # Extend the current value and recompute logits
        curr_value += curr_token
        input_ids = llm._encode(ins + curr_value).tolist()[0]
        logits = llm.get_logits_from_input_ids(input_ids)

    return {p_name: curr_value}


def generate_bool(llm: LLM_Model, ins: str, p_name: str) -> dict[str, bool]:
    return {"is": True}


"""
Type alias for generator functions.
Each generator takes:
- an LLM model
- an instruction string
- a parameter name
And returns a dictionary with the generated value.
"""
GeneratorFn = Callable[[LLM_Model, str, str], dict[str, Any]]

"""
Mapping between parameter types and their corresponding generator functions.
This is used to select the correct generator based on the parameter type.
"""
TYPE_GENERATORS: dict[str, GeneratorFn] = {
    "number": generate_number,
    "string": generate_string,
    "bool":   generate_bool,
}


def generate_parameters(
    model: LLM_Model, func: FuncDef, prompt: str, instructions: str
) -> dict[str, Any]:
    """Generate all argument values for a function call.

    Builds instructions progressively — each generated value is appended
    to the context before generating the next parameter.

    Args:
        model: The LLM wrapper used to score tokens.
        func: The target function definition, including parameter schema.
        prompt: Natural language description of the desired operation.
        instructions: Base instruction prompt from get_base_instructions.

    Returns:
        A dict mapping every parameter name to its decoded value.

    Raises:
        ValueError: If a parameter type has no registered generator.
    """
    res: dict[str, Any] = {}

    for param_name, param in func.parameters.items():
        instructions += f'{param_name}='
        generator = TYPE_GENERATORS.get(param.type)
        if generator is None:
            raise ValueError(
                f"No generator registered for type '{param.type}' "
                f"(parameter '{param_name}' of '{func.name}')"
            )
        value = generator(model, instructions, param_name)
        res.update(value)
        # Accumulate generated value so the next parameter has context
        instructions += f'{list(value.values())[0]}\n'

    return res
