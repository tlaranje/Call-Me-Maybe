from llm_sdk import Small_LLM_Model
import json


def build_prompt(functions: list, prompt: str) -> str:
    functions_json = json.dumps([{
                    "name": f.name,
                    "description": f.description,
                    "parameters": {
                        k: {"type": v.type}
                        for k, v in f.parameters.items()
                    }} for f in functions], indent=2
    )
    return (
        f"Functions: \n{functions_json}\n"
        f"Prompt: \"{prompt}\"\n"
        f"Respond ONLY with a JSON object in the format: \n"
        '{"name": "<function_name>", "parameters": {<key>: <value>}}\n'
    )


def load_vocab(model: Small_LLM_Model) -> dict[int, str]:
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    id_to_token = {v: k for k, v in vocab.items()}
    return id_to_token


def get_valid_token_ids(
    vocab: dict[int, str], js: str, functions: list
) -> list[int]:
    return []


def generate_json(model, input_ids, valid_tokens_fn) -> None:
    pass
