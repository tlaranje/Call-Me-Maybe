from src.constants import Colors as C
from pydantic import ValidationError
from llm_sdk import Small_LLM_Model
from src.function_selector import generate_function_name
from src.param_generator import generate_parameters, get_params_instructions
from src.validation_models import load_and_validate, FunctionDefinition
from src.get_args import get_args
import json
import os
from typing import Any


def write_results(file: str, res: list[dict[str, Any]]) -> None:
    """
    Write the results to a JSON file.

    Args:
        file (str): Path to the output file.
        res (list): List of results to write.
    """
    os.makedirs("data/output", exist_ok=True)
    with open(file, "w") as fd:
        fd.write(json.dumps(res, indent=4))


def main() -> None:
    """
    Entry point of the function calling pipeline.

    Loads the LLM, reads the input prompts and function definitions,
    and for each prompt selects the correct function using constrained
    decoding, then extracts its arguments. Results are written
    incrementally to the output JSON file after each prompt.
    """
    final_json: list[dict[str, Any]] = []

    try:
        llm = Small_LLM_Model()
        args = get_args()
        data = load_and_validate(args)
        print()

        print(
            f"{C.BLUE}=== Generating functions "
            f"names and parameters ==={C.END}\n"
        )

        functions = data['functions']
        for p in data['prompts']:
            print(f"{C.GREEN}\"prompt\": \"{p.prompt}\"{C.END}")
            # Generate the best matching function name for the prompt
            func_name = generate_function_name(llm, functions, p.prompt)

            # Find the full function definition from the list
            func: FunctionDefinition | None = next(
                (f for f in functions if f.name == func_name), None
            )

            if func is None:
                raise ValueError(f"Function '{func_name}' not found")

            # Build the base instruction prompt for parameter extraction
            instructions = get_params_instructions(func, p.prompt)

            # Generate parameters only if the function expects them
            params = {}
            if func and func.parameters:
                params = generate_parameters(
                    llm, func, p.prompt, instructions
                )

            # Append the result for this prompt to the output list
            final_json.append({
                "prompt": p.prompt,
                "name": func_name,
                "parameters": params
            })

            # Write incrementally so partial results are saved on crash
            write_results(args['output'], final_json)
            print()

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
