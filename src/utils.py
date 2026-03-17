from llm_sdk import Small_LLM_Model as LLM_Model
import json
from typing import Any


def load_vocab(model: LLM_Model) -> Any:
    vocab_path = model.get_path_to_vocabulary_json()
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    return vocab
