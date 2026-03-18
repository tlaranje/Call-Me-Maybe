from src.constants import Colors as C
from pydantic import ValidationError
from llm_sdk import Small_LLM_Model
from src.function_selector import generate_function_name
from src.param_generator import generate_parameters
from src.validation_models import load_and_validate
from src.get_args import get_args
import json


def write_results(file: str, res: list) -> None:
    with open(file, "w") as fd:
        fd.write(json.dumps(res, indent=2))


def main() -> None:
    try:
        llm = Small_LLM_Model()
        print()
        args = get_args()
        data = load_and_validate(args)

        functions = data['functions']
        for p in data['prompts']:
            func_name = generate_function_name(llm, functions, p.prompt)
            func = next((f for f in functions if f.name == func_name), None)
            params = {}
            if func and func.parameters:
                params = generate_parameters(llm, func, p.prompt)
                print(p.prompt, "->", params)
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
