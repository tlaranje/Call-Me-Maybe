from llm_sdk import Small_LLM_Model as LLM_Model
import json
from typing import Any


def load_vocab(llm: LLM_Model) -> Any:
    """
    Load the vocabulary JSON file associated with the given LLM model.

    Args:
        model (LLM_Model): The language model providing the vocabulary path.

    Returns:
        Any: The parsed vocabulary mapping token IDs to token strings.
    """
    vocab_path = llm.get_path_to_vocab_file()
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    return vocab
