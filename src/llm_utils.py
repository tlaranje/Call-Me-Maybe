from llm_sdk import Small_LLM_Model
from enum import Enum
import json


class State(Enum):
    # '{"name":"fn_add_numbers","parameters":{"a":2.0,"b":3.0}}'

    START = 0              # {
    NAME_KEY = 1           # "name"
    NAME_COLON = 2         # :
    NAME_OPEN = 3          # "
    NAME_VALUE = 4         # fn_add_numbers
    NAME_CLOSE = 5         # "
    COMMA = 6              # ,
    PARAMS_KEY = 7         # "parameters"
    PARAMS_COLON = 8       # :
    PARAMS_OPEN = 9        # {
    PARAM_NAME_OPEN = 10   # "
    PARAM_NAME = 11        # a
    PARAM_NAME_CLOSE = 12  # "
    PARAM_COLON = 13       # :
    PARAM_VALUE = 14       # 2.0
    PARAM_COMMA = 15       # ,
    PARAMS_CLOSE = 16      # }
    END = 17               # }


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
    js = '{"name":"fn_add_numbers","parameters":{"a":'
    print(js)
    f_name = js.split('{"name":"')[1].split('"')[0]
    f_names = [f.name for f in functions]
    fn = next((f for f in functions if f.name == f_name), None)

    def is_param_comma(js: str) -> bool:
        jss = js.split('parameters":')[1]
        filled_p, remaining_p = check_params(jss, fn.parameters)
        if not fn or '"parameters":{' not in js or len(remaining_p) != 1:
            return False
        inner = js.split('"parameters":{')[1]
        return (
            ':' in inner
            and inner.count('.') == 1 or inner.count('"') == 2
            and not inner.endswith('.') or inner.endswith('"')
        )

    def is_params_close(js: str) -> bool:
        if not fn or '"parameters":{' not in js:
            return False
        part_2 = js.split('"parameters":')[1]
        filled_p, remaining_p = check_params(part_2, fn.parameters)
        return len(remaining_p) == 0

    states = {
        State.START: lambda js: js == '',

        State.NAME_KEY: lambda js: js != '' and '{"name'.startswith(js),

        State.NAME_COLON: lambda js: (
            js.startswith('{"name"') and not js.startswith('{"name":')
        ),

        State.NAME_OPEN: lambda js: (
            js.startswith('{"name":') and not js.startswith('{"name":"')
        ),

        State.NAME_VALUE: lambda js: (
            '{"name":"' in js
            and f_name not in f_names
            and any(f.startswith(f_name) for f in f_names)
        ),

        State.NAME_CLOSE: lambda js: (
            '{"name":"' in js
            and f_name in f_names
            and not (',' in js)
            and not js.endswith('"')
        ),

        State.COMMA: lambda js: (
            f_name in f_names
            and js == '{"name":"' + f_name + '"'
        ),

        State.PARAMS_KEY: lambda js: (
            '{"name":"' + f_name + '",' in js
            and '"parameters'.startswith(
                js.split('{"name":"' + f_name + '",')[1]
            )
        ),

        State.PARAMS_COLON: lambda js: (
            '"parameters"' in js
            and not js.split('"parameters"')[1].startswith(':')
        ),

        State.PARAMS_OPEN: lambda js: (
            '"parameters":' in js
            and not js.split('"parameters":')[1].startswith('{')
        ),

        State.PARAM_NAME_OPEN: lambda js: (
            '"parameters":{' in js
            and js.split('"parameters":{')[1] == ''
        ),

        State.PARAM_NAME: lambda js: (
            '"parameters":{"' in js
            and any(
                p.startswith(js.split('"parameters":{"')[1])
                for p in fn.parameters
            )
        ),

        State.PARAM_COLON: lambda js: (
            any(
                js.split('"parameters":{')[1] == f'"{p}"'
                for p in fn.parameters
            )
        ),

        State.PARAM_VALUE: lambda js: (
            '"parameters":{' in js
            and ':' in js.split('"parameters":{')[1]
            and not js.split('"parameters":{')[1].endswith(':')
        ),

        State.PARAM_NAME_CLOSE: lambda js: (
            fn is not None and '"parameters":{"' in js
            and not js.split('"parameters":{')[1].endswith(':')
            and ':' not in js.split('"parameters":{')[1]
            and any(
                js.split('"parameters":{"')[1].split('"')[0] == p
                for p in fn.parameters
            )
        ),

        State.PARAM_COMMA: is_param_comma,

        State.PARAMS_CLOSE: is_params_close,
    }

    for state, condition in states.items():
        if condition(js):
            return state
    return State.END


"""
    part_1 = js.split('"parameters"')[1] if '"parameters"' in js else ""
    part_2 = js.split('"parameters":')[1] if '"parameters"' in js else ""
    generated = js.split(",")[1] if "," in js else ""

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
