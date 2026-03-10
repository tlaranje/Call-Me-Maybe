from llm_sdk import Small_LLM_Model
from enum import Enum
import json


class State(Enum):
    START = 0
    NAME_KEY = 1
    NAME_COLON = 2
    NAME_QUOTE = 3
    NAME_VALUE = 4
    NAME_CLOSE = 5
    COMMA = 6
    PARAMS_KEY = 7
    PARAMS_COLON = 8
    PARAMS_OPEN = 9
    PARAM_NAME = 10
    PARAM_COLON = 11
    PARAM_VALUE = 12
    PARAM_COMMA = 13
    PARAMS_CLOSE = 14
    END = 15


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


def get_state(partial_json: str, functions: list) -> str:
    return ""


def check_params(js: str, fn_params: dict) -> tuple[list[str], list[str]]:
    filled_p = []
    remaining_p = []
    in_string = False
    start = 1

    for i, c in enumerate(js):
        if c == '"':
            in_string = not in_string
        elif c == ',' and not in_string:
            filled_p.append(js[start:i].strip().split(':')[0].strip())
            start = i + 1
        elif c == '}' and not in_string:
            if start < i:
                filled_p.append(js[start:i].strip().split(':')[0].strip())
            break

    remaining_p = [p for p in fn_params if f'"{p}"' not in filled_p]
    return (filled_p, remaining_p)


def get_valid_token_ids(
    vocab: dict[int, str], js: str, functions: list
) -> list[int]:
    res: list[int] = []
    name = js.split('{"name":"')[1].split('"')[0] if '{"name":"' in js else ""
    fn_names = [f.name for f in functions]
    generated = js.split(',')[1] if ',' in js else ""
    part_1 = js.split('"parameters"')[1] if '"parameters"' in js else ""
    part_2 = js.split('"parameters":')[1] if '"parameters"' in js else ""
    fn = next((f for f in functions if f.name == name), None)

    for t_id, t_str in vocab.items():
        t_str = t_str.strip()

        if js == "":
            if t_str == "{":
                res.append(t_id)
        elif js == "{":
            if '"name"'.startswith(t_str):
                res.append(t_id)
        elif '{"name"' in js and ':' not in js:
            if t_str == ':':
                res.append(t_id)
        elif '{"name":' in js and '"' not in js.split(':')[1]:
            if t_str == '"':
                res.append(t_id)
        elif '{"name":"' in js and '"' not in js.split('{"name":"')[1]:
            if any(name == f for f in fn_names):
                if t_str == '"':
                    res.append(t_id)
            elif any(f.startswith(name + t_str) for f in fn_names):
                res.append(t_id)
        elif '{"name":"' + name in js and "," not in js:
            if t_str == ',':
                res.append(t_id)
        elif '"parameters"'.startswith(generated + t_str):
            res.append(t_id)
        elif '"parameters"' in js and ':' not in part_1:
            if t_str == ':':
                res.append(t_id)
        elif ':' in part_1 and '{' not in part_1:
            if t_str == '{':
                res.append(t_id)
        elif '"parameters":' in js and '{' in part_2:
            filled_p, remaining_p = check_params(part_2, fn.parameters)
            """
            params_gerados = extrair as chaves que já aparecem no js depois
            de "parameters":{"
            params_restantes = todos os params da função menos os já gerados
            se qualquer prefixo de '"param_restante"' coincide com
            t_str → válido
            """
            """ if fn:
                params = fn.parameters
                if any(f'"{p}"'.startswith(t_str) for p in params):
                    res.append(t_id) """

    return res


def generate_json(model, input_ids, valid_tokens_fn) -> None:
    pass
