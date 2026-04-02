from src.models import load_and_validate, FunctionDefinition
from src.generators import FunctionCaller
from src.utils import render_panel
from src.parse_args import parse_args

from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich import print

from pydantic import ValidationError
from llm_sdk import Small_LLM_Model
from typing import Any
import json
import os


def write_results(file: str, res: list[dict[str, Any]]) -> None:
    """
    Write results to a JSON file.

    Ensures the output directory exists and saves the results
    with indentation for readability.

    Args:
        file (str): Path to the output JSON file.
        res (list[dict[str, Any]]): List of results.

    Returns:
        None
    """
    # Ensure output directory exists
    os.makedirs("data/output", exist_ok=True)

    # Write formatted JSON
    with open(file, "w") as fd:
        fd.write(json.dumps(res, indent=4))


def main() -> None:
    """
    Entry point for the function-calling pipeline.

    This function orchestrates the full workflow:
    - Loads and validates input data
    - Generates function names using the LLM
    - Generates parameters for each function
    - Streams progress to the terminal using Rich
    - Saves results incrementally to a JSON file

    Args:
        None

    Returns:
        None
    """
    # Accumulates all results across prompts
    final_json: list[dict[str, Any]] = []

    try:
        # Initialize model and load input data
        llm = Small_LLM_Model()
        args = parse_args()
        data = load_and_validate(args)

        print()
        print(
            "[bold blue]=== Generating functions "
            "names and parameters ===[/bold blue]\n"
        )

        functions = data['functions']
        console = Console()

        # Live UI for streaming updates (animation)
        with Live(console=console, refresh_per_second=20) as live:
            for p in data['prompts']:
                caller = FunctionCaller()
                # 1. Initial state
                live.update(render_panel(p.prompt, None, None))

                # 2. Function selection
                func_name = caller.function_generator.generate_function_name(
                    llm, functions, p.prompt, live
                )

                live.update(render_panel(p.prompt, func_name, None))

                # 3. Resolve function definition
                func: FunctionDefinition | None = next(
                    (f for f in functions if f.name == func_name), None
                )

                if func is None:
                    raise ValueError(f"Function '{func_name}' not found")

                # 4. Parameter generation
                instructions = caller.get_params_instructions(func, p.prompt)

                params: dict[str, Any] = {}

                if func.parameters:
                    params = caller.generate_parameters(
                        llm, func, p.prompt, instructions, live
                    )

                # live.update(render_panel(p.prompt, func_name, params))

                # 5. Save and write result
                final_json.append({
                    "prompt": p.prompt,
                    "name": func_name,
                    "parameters": params
                })

                write_results(args['output'], final_json)

                # Print stable result below animated panel
                live.update("")
                live.console.print(
                    render_panel(p.prompt, func_name, params)
                )
                break
    except ValidationError as e:
        # Handle structured validation errors
        for error in e.errors():
            msg = error['msg'].removeprefix("Value error, ")
            print(Panel(
                f"[bold red]{msg}[/bold red]",
                title="[bold white]Validation Error[/bold white]"
            ))
        exit()

    except Exception as e:
        # Catch-all for unexpected runtime errors
        print(Panel(
            f"[bold red]{e}[/bold red]",
            title="[bold white]Error[/bold white]"
        ))
        exit()


if __name__ == "__main__":
    main()
