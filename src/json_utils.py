from typing import Any
import json


def load_json_file(path: str) -> Any:
    try:
        with open(path, "r") as fd:
            return json.load(fd)
    except FileNotFoundError:
        raise FileNotFoundError("File not found!")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format! - {path}")


"""
def get_state(js: str, functions: list) -> str:
    name = js.split('{"name":"')[1].split('"')[0] if '{"name":"' in js else ""
    fn_names = [f.name for f in functions]

    if js == "":
        return State.START

    if js == "{":
        return State.NAME_KEY

    if '{"name"' in js and ':' not in js:
        return State.NAME_COLON

    if '{"name":' in js and '"' not in js.split(':')[1]:
        return State.NAME_QUOTE

    if '{"name":"' in js and '"' not in js.split('{"name":"')[1]:
        if any(name == f for f in fn_names):
            return State.NAME_CLOSE
        return State.NAME_VALUE

    if '{"name":"' + name in js and "," not in js:
        return State.COMMA

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


def get_valid_token_ids(
    vocab: dict[int, str], js: str, functions: list
) -> list[int]:
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
