from llm_sdk import Small_LLM_Model as LLM_Model
from src.utils import load_vocab


def get_func_instructions(funcs: list, prompt: str) -> str:
    """
    Create instructions for the LLM to choose the best function
    based on a prompt.

    Args:
        funcs (list): List of available functions.
        prompt (str): User's prompt in natural language.

    Returns:
        str: Formatted instruction string for the LLM.
    """
    return (
        "<|im_start|>system\n"
        "You are a function selector.\n"
        "Choose the best function based on the user prompt.\n"
        "Return ONLY the function name.\n"
        "IMPORTANT RULES:\n"
        "- 'reverse' means flipping text order\n"
        "- 'replace', 'substitute', 'change' means modifying parts of text\n"
        "- 'numbers', 'vowels', 'words' → use substitution function\n\n"
        + str(funcs) + "\n<|im_end|>\n"
        "<|im_start|>user\n" + prompt + "\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def generate_function_name(llm: LLM_Model, funcs: list, prompt: str) -> str:
    """
    Pick the best matching function name for a given prompt.

    Implements a greedy constrained decoding loop: at each step, the
    vocabulary is filtered to only tokens that extend the current prefix
    toward at least one valid function name. The highest-scoring surviving
    token is appended, and the process repeats until the prefix matches
    exactly one function name or no valid extension exists.

    Args:
        llm (LLM_Model): The language model used to generate tokens.
        funcs (list): List of available functions.
        prompt (str): User's prompt in natural language.

    Returns:
        The selected function name (e.g. `"fn_add_numbers"`), or an
        empty string if constrained decoding could not resolve any
        valid function from the given prompt.
    """
    f_names = sorted([f.name for f in funcs])
    output = "fn_"
    instructions = get_func_instructions(funcs, prompt)
    vocab = load_vocab(llm)

    def encode_and_get_logits() -> tuple:
        """
        Re-encode the current context and return fresh logits.

        Concatenates the system instructions with the partially built
        function name so the model scores the next token in context.

        Returns:
            A tuple of (input_ids, logits) where input_ids is the list
            of token IDs for the full context and logits is a 1-D tensor
            of raw scores over the entire vocabulary.
        """
        ids = llm._encode(instructions + output).tolist()[0]
        return ids, llm.get_logits_from_input_ids(ids)

    # Get initial logits
    input_ids, logits = encode_and_get_logits()

    # Keep generating until the output matches a full function name
    while output not in f_names:
        # Filter to only functions that still match the current prefix
        ft_list = [f for f in f_names if f.startswith(output)]

        # If only one function matches, we can finish immediately
        if len(ft_list) == 1:
            output = ft_list[0]
            break

        # If no function matches the prefix, decoding failed
        if not ft_list:
            return ""

        # Collect tokens that keep the prefix valid
        valid_tokens = []
        for s, tid in vocab.items():
            # Remove leading space marker if present
            token_str = s[1:] if s.startswith('Ġ') else s

            # Token must not be empty and must extend toward a valid function
            if token_str and any(
                f.startswith(output + token_str) for f in f_names
            ):
                valid_tokens.append((tid, token_str))

        # If no valid token can extend the prefix, decoding fails
        if not valid_tokens:
            return ""

        # Pick the valid token with the highest score
        token_id, token_str = max(valid_tokens, key=lambda x: logits[x[0]])

        # Append the token to the output and update logits
        output += token_str
        input_ids, logits = encode_and_get_logits()

    return output
