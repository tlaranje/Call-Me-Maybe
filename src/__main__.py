from src.constants import Colors as C
from pydantic import ValidationError
from src.llm_utils import get_valid_token_ids, load_vocab, get_state
from llm_sdk import Small_LLM_Model
from src.validation_models import load_and_validate
from src.get_args import get_args
import json


def write_results(file: str, res: list) -> None:
    with open(file, "w") as fd:
        fd.write(json.dumps(res, indent=2))


def main() -> None:
    try:
        # model = Small_LLM_Model()
        # print()
        args = get_args()
        data = load_and_validate(args)

        functions = data['functions']
        # vocab = load_vocab(model)
        # js = '{"name":"fn_add_numbers","parameters":{"a":2.0,"b":3.0}}'
        js = '{"name":'
        print(get_state(js, functions))

        """ lst_ints = get_valid_token_ids(vocab, js, functions)
        for token_id in lst_ints:
            print(repr(token_id), repr(vocab[token_id])) """
    except ValidationError as e:
        for error in e.errors():
            msg = error['msg'].removeprefix("Value error, ")
            print(f"{C.RED}{msg}{C.END}")
        exit()
    except Exception as e:
        print(f"{C.RED}{e}{C.END}")
        exit()


if __name__ == "__main__":
    main()
