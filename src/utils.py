from llm_sdk import Small_LLM_Model as LLM_Model
import json
from typing import Any


def load_vocab(model: LLM_Model) -> Any:
    vocab_path = model.get_path_to_vocabulary_json()
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    return vocab


""" def generate_number(
    model: LLM_Model, func: FuncDef, prompt: str
) -> dict[str, int]:
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
    return output """