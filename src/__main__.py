import json
import os
from typing import Any

from llm_sdk import Small_LLM_Model
from pydantic import ValidationError
from rich import print
from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from src.generators import FunctionCaller
from src.models import load_and_validate
from src.parse_args import parse_args
from src.utils import animate_field, render_panel


def write_results(file: str, res: list[dict[str, Any]]) -> None:
    """
    Writes results to a JSON file.

    Args:
        file (str): Path to the output JSON file.
        res (list[dict[str, Any]]): List of results.
    """
    output_dir = os.path.dirname(file) or "data/output"
    os.makedirs(output_dir, exist_ok=True)

    with open(file, "w", encoding="utf-8") as fd:
        json.dump(res, fd, indent=4, ensure_ascii=False)


def main() -> None:
    """
    Entry point for the function-calling pipeline.

    Orchestrates loading, generation via LLM, and real-time UI updates.
    """
    final_json: list[dict[str, Any]] = []

    try:
        llm = Small_LLM_Model()
        args = parse_args()
        data = load_and_validate(args)
        functions = data['functions']
        console = Console()

        print("\n[bold blue]=== Generating functions ===[/bold blue]\n")

        with Live(console=console, refresh_per_second=20) as live:
            for p in data['prompts']:
                caller = FunctionCaller()

                if p.prompt == "":
                    live.console.print(Panel.fit(
                        "[bold red] Prompt cannot be empty [/bold red]",
                        title="Error"
                    ))
                    continue

                # 1. Name generation
                func_name = caller.function_generator.generate(
                    llm, functions, p.prompt
                )
                animate_field(
                    live, p.prompt, func_name, field="func", new_text=func_name
                )

                if not func_name or func_name == "fn_none":
                    live.console.print(Panel.fit(
                        f"[bold red]No function for:[/bold red] {p.prompt}",
                        title="Error"
                    ))
                    continue

                # 2. Definition lookup
                func = next(
                    (f for f in functions if f.name == func_name), None
                )

                if func is None:
                    live.console.print(Panel.fit(
                        f"[bold red]Error:[/bold red] '{func_name}' not found."
                    ))
                    continue

                # 3. Parameter generation
                params = {}
                if func.parameters:
                    params = caller.generate_parameters(llm, func, p.prompt)

                is_valid = not any(v is None for v in params.values())
                params_text = (
                    str(params) if is_valid else "[bold red]Invalid "
                    "Value[/bold red]"
                )

                animate_field(
                    live,
                    prompt=p.prompt,
                    func_name=func_name,
                    field="params",
                    new_text=params_text,
                    animate=is_valid
                )

                # 4. Save results
                if is_valid:
                    final_json.append({
                        "prompt": p.prompt,
                        "name": func_name,
                        "parameters": params
                    })
                    write_results(args['output'], final_json)

                live.update("")
                live.console.print(
                    render_panel(p.prompt, func_name, params_text)
                )

    except ValidationError as e:
        for error in e.errors():
            msg = error['msg'].replace("Value error, ", "")
            print(Panel(f"[bold red]{msg}[/bold red]", title="Validation"))
    except Exception as e:
        print(Panel.fit(f"[bold red]{e}[/bold red]", title="Error"))


if __name__ == "__main__":
    main()
