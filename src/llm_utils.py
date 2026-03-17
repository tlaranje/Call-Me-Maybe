from llm_sdk import Small_LLM_Model as LLM_Model
from src.validation_models import FunctionDefinition as FuncDef
import json


def generate_numbers(
    model: LLM_Model, func: FuncDef, prompt: str
) -> dict[str, int]:
    output = {}
    instructions = (
        f'<|im_start|>system\n'
        f'You are a params generate assistant.<|im_end|>\n'
        f'<|im_start|>user\n'
        f'In this function: {func}\n'
        f'Generate the params values '
        f'for this prompt is: "{prompt}"?<|im_end|>\n'
        f'<|im_start|>assistant\n'
        f'str:int'
    )
    vocab = load_vocab(model)
    for param_name, param in func.parameters.items():
        current_value = ""
        while True:
            input_ids = model._encode(instructions + current_value).tolist()[0]
            logits = model.get_logits_from_input_ids(input_ids)
            token_id = logits.index(max(logits))
            token_str = vocab[token_id]
            print(token_str)
            print(output)
            if token_str in [",", "}"]:
                break
            try:
                float(current_value + token_str)
                current_value += token_str
            except ValueError:
                pass
        output[param_name] = float(current_value)
    return output


def load_vocab(model: LLM_Model) -> any:
    vocab_path = model.get_path_to_vocabulary_json()
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    return vocab


def get_func_instructions(funcs: list, prompt: str) -> str:
    functions = [
        {
            "name": f.name,
            "description": f.description
        }
        for f in funcs
    ]

    return (
        "<|im_start|>system\n"
        "You are a function selector.\n"
        "Choose the best function based on the user request.\n"
        "Return ONLY the function name.\n"
        "IMPORTANT RULES:\n"
        "- 'reverse' means flipping text order\n"
        "- 'replace', 'substitute', 'change' means modifying parts of text\n"
        "- 'numbers', 'vowels', 'words' → use substitution function\n\n"
        + json.dumps(functions)
        + "\n<|im_end|>\n"
        "<|im_start|>user\n"
        + prompt +
        "\n<|im_end|>\n"
        "<|im_start|>assistant\n"
        "Examples:\n"
        "Replace numbers → fn_substitute_string_with_regex\n"
        "Replace vowels → fn_substitute_string_with_regex\n"
        "Reverse string → fn_reverse_string\n\n"
    )


def generate_function_name(
    model: LLM_Model, funcs: list, prompt: str
) -> str:
    """Select the best matching function name for a given prompt.

    Uses the model's token logits to greedily build a function name
    character-by-character, constrained to only the available function
    names. At each step, only tokens that keep at least one valid
    function name reachable are considered.

    Parameters
    ----------
    model : LLM_Model
        The language model used to score tokens.
    funcs : list
        List of function objects, each with a `.name` attribute.
    prompt : str
        Natural language description of the desired operation.

    Returns
    -------
    str
        The name of the selected function, or None if no valid
        function could be resolved.
    """
    # Sort function names to ensure deterministic behaviour
    f_names = sorted([f.name for f in funcs])
    output = "fn_"  # all function names share this prefix
    instructions = get_func_instructions(funcs, prompt)
    vocab = load_vocab(model)

    def encode_and_get_logits():
        # Encode the full context (instructions + current output so far)
        # and return the token ids and next-token logits
        ids = model._encode(instructions + output).tolist()[0]
        return ids, model.get_logits_from_input_ids(ids)

    input_ids, logits = encode_and_get_logits()

    while output not in f_names:
        # Narrow down to functions still reachable given current output
        ft_list = [f for f in f_names if f.startswith(output)]
        if len(ft_list) == 1:
            # Only one candidate left — no need to consult the model
            output = ft_list[0]
            break

        # Build a list of tokens that would keep at least one
        # function name reachable if appended to the current output
        valid_tokens = []

        for s, tid in vocab.items():
            # Strip the leading space marker used by some tokenizers
            token_str = s[1:] if s.startswith('Ġ') else s

            if token_str and any(
             f.startswith(output + token_str) for f in f_names):
                valid_tokens.append((tid, token_str))

        if not valid_tokens:
            # No token can extend output towards a valid function name
            return None

        # Pick the valid token with the highest logit score
        token_id, token_str = max(valid_tokens, key=lambda x: logits[x[0]])

        output += token_str
        input_ids, logits = encode_and_get_logits()

    return output
