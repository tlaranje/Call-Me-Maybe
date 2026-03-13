from llm_sdk import Small_LLM_Model as LLM_Model
from src.validation_models import FunctionDefinition as FuncDef
import json
import math


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


def load_vocab(model: LLM_Model) -> dict[int, str]:
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
            input_ids = model.encode(instructions + current_value).tolist()[0]
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


def generate_function_name(model: LLM_Model, funcs: list, prompt: str) -> str:
    f_names = [f.name for f in funcs]
    output = "fn_"
    instructions = (
        f'<|im_start|>system\n'
        f'You are a function calling assistant.<|im_end|>\n'
        f'<|im_start|>user\n'
        f'Available functions: {f_names}\n'
        f'The most appropriate function '
        f'for this prompt is: "{prompt}"?<|im_end|>\n'
        f'<|im_start|>assistant\n'
        f'fn_'
    )
    input_ids = model.encode(instructions + output).tolist()[0]
    logits = model.get_logits_from_input_ids(input_ids)
    vocab = load_vocab(model)
    while output not in f_names:
        token_id = logits.index(max(logits))
        token_str = vocab[token_id].replace("Ġ", " ").strip()

        for f in f_names:
            if f.startswith(output + token_str):
                output += token_str
                input_ids = model.encode(instructions + output).tolist()[0]
                logits = model.get_logits_from_input_ids(input_ids)
                break
            else:
                logits[token_id] = -math.inf
    return output
