from llm_sdk import Small_LLM_Model
from enum import Enum
import json


class State(Enum):
    # '{"name":"fn_add_numbers","parameters":{"a":2.0,"b":3.0}}'

    START = 0          # {
    NAME_KEY = 1       # "name"
    NAME_COLON = 2     # :
    NAME_VALUE = 4     # "fn_add_number"
    COMMA = 6          # ,
    PARAMS_KEY = 7     # "parameters"
    PARAMS_COLON = 8   # :
    PARAMS_OPEN = 9    # {
    PARAM_NAME = 10    # "a"
    PARAM_COLON = 11   # :
    PARAM_VALUE = 12   # 2.0
    PARAM_COMMA = 13   # , if have more that one parameter
    PARAMS_CLOSE = 14  # }
    END = 15           # }


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


def get_state(js: str, functions: list) -> str:
    f_name = js.split('{"name":')[1].split(',')[0] if '{"name":' in js else ""
    f_names = [f'"{f.name}"' for f in functions]

    def is_params_key(js):
        generated = js.split(",")[1] if "," in js else ""
        return ('"parameters"'.startswith(generated))

    def is_params_colon(js: str) -> bool:
        gen = js.split(",")[1] if "," in js else ""
        return '"parameters"'.startswith(gen) and '"parameters"' not in js

    def is_params_open(js: str) -> bool:
        generated = js.split(",")[1] if "," in js else ""
        return (
            '"parameters"'.startswith(generated) and '"parameters"' not in js
        )

    states = {
        State.START: lambda js: js in '{',
        State.NAME_KEY: lambda js: js != "" and '{"name"'.startswith(js),
        State.NAME_COLON: lambda js: js.endswith(':'),
        State.NAME_VALUE: lambda js: (
            '{"name":' in js and '"' not in js.split('{"name":')[1]
        ),
        State.COMMA: lambda js: (
            '{"name":' + f_name + ',' in js and js.endswith(',')
        ),
        # State.PARAMS_KEY: is_params_key
    }
    """ for state, condition in states.items():
        if condition(js):
            return state """
    js_full = '{"name":"fn_add_numbers","parameters":{"a":2.0,"b":3.0}}'

    for i in range(1, len(js_full) + 1):
        js = js_full[:i]
        matched = next((s for s, cond in states.items() if cond(js)), None)
        print(f"{matched!s:<20} → {js}")


"""
    part_1 = js.split('"parameters"')[1] if '"parameters"' in js else ""
    part_2 = js.split('"parameters":')[1] if '"parameters"' in js else ""
    generated = js.split(",")[1] if "," in js else ""

    if '"parameters"'.startswith(generated + next(iter(generated), "")):
        if '"parameters"' not in js:
            return State.PARAMS_KEY

    if '"parameters"' in js and ":" not in part_1:
        return State.PARAMS_COLON

    if ":" in part_1 and "{" not in part_1:
        return State.PARAMS_OPEN

    if js.endswith("}}"):
        return State.END

    if '"parameters":' in js and "{" in part_2:
        fn = next((f for f in functions if f.name == name), None)
        if fn:
            filled_p, remaining_p = check_params(part_2, fn.parameters)
            if remaining_p:
                inner = part_2.lstrip("{").rstrip("}")
                last_comma = inner.rfind(",")
                segment = inner[last_comma + 1:] if last_comma != -1 else inner

                if ":" not in segment:
                    return (
                        State.PARAM_NAME
                        if segment.strip() else State.PARAM_COMMA
                    )
                elif segment.count(":") == 1:
                    return (
                        State.PARAM_COLON
                        if segment.endswith(":") else State.PARAM_VALUE
                    )
                return State.PARAM_VALUE
            else:
                return State.PARAMS_CLOSE

    return State.START
    res: list[int] = []
    state = get_state(js, functions)
    name = js.split('{"name":"')[1].split('"')[0] if '{"name":"' in js else ""
    fn_names = [f.name for f in functions]
    generated = js.split(',')[1] if ',' in js else ""
    part_2 = js.split('"parameters":')[1] if '"parameters"' in js else ""
    fn = next((f for f in functions if f.name == name), None)

    for t_id, t_str in vocab.items():
        t_str = t_str.strip()

        if state == State.START:
            if t_str == "{":
                res.append(t_id)

        elif state == State.NAME_KEY:
            if '"name"'.startswith(t_str):
                res.append(t_id)

        elif state == State.NAME_COLON:
            if t_str == ':':
                res.append(t_id)

        elif state == State.NAME_QUOTE:
            if t_str == '"':
                res.append(t_id)

        elif state == State.NAME_VALUE:
            if any(f.startswith(name + t_str) for f in fn_names):
                res.append(t_id)

        elif state == State.NAME_CLOSE:
            if t_str == '"':
                res.append(t_id)

        elif state == State.COMMA:
            if t_str == ',':
                res.append(t_id)

        elif state == State.PARAMS_KEY:
            if '"parameters"'.startswith(generated + t_str):
                res.append(t_id)

        elif state == State.PARAMS_COLON:
            if t_str == ':':
                res.append(t_id)

        elif state == State.PARAMS_OPEN:
            if t_str == '{':
                res.append(t_id)

        elif state in (
            State.PARAM_NAME,
            State.PARAM_COMMA,
            State.PARAM_COLON,
            State.PARAM_VALUE
        ):
            if fn:
                filled_p, remaining_p = check_params(part_2, fn.parameters)
                print(filled_p, remaining_p)
                if state == State.PARAM_NAME:
                    if any(f'"{p}"'.startswith(t_str) for p in remaining_p):
                        res.append(t_id)
                elif state == State.PARAM_COMMA:
                    if t_str == ',':
                        res.append(t_id)
                elif state == State.PARAM_COLON:
                    if t_str == ':':
                        res.append(t_id)

        elif state == State.PARAMS_CLOSE:
            if t_str == '}':
                res.append(t_id)

        elif state == State.END:
            pass

    return res
"""


def get_valid_token_ids(
    vocab: dict[int, str], js: str, functions: list
) -> list[int]:
    return []


def generate_json(model, input_ids, valid_tokens_fn) -> None:
    pass
