from llm_sdk import Small_LLM_Model as LLM_Model
from rich.markup import escape
from rich.panel import Panel
from typing import Any
import json


def render_panel(
    prompt: str,
    func_name: str | None = None,
    params: dict[Any, Any] | None = None
) -> Panel:
    """
    Create a formatted Rich panel displaying the current pipeline state.

    Args:
        prompt (str): User input prompt.
        func_name (str | None): Generated function name.
        params (dict | None): Generated parameters.

    Returns:
        Panel: Configured Rich Panel ready for rendering.
    """
    safe_prompt = escape(prompt)

    safe_func = escape(func_name) if func_name else "..."

    safe_params: Any
    if isinstance(params, dict):
        params_str = json.dumps(params, ensure_ascii=False)
        safe_params = escape(params_str)
    else:
        safe_params = "..."

    return Panel.fit(
        f'[bold yellow]Prompt:[/bold yellow] {safe_prompt}\n'
        f'[bold green]Function:[/bold green] {safe_func}\n'
        f'[bold cyan]Parameters:[/bold cyan] {safe_params}',
        title="[bold white]Result[/bold white]",
        width=200
    )


def load_vocab(llm: LLM_Model) -> Any:
    """
    Load the vocabulary associated with the given LLM model.

    Reads the vocabulary JSON file from the model path and parses it
    into a dictionary mapping tokens to IDs.

    Args:
        llm (LLM_Model): Language model providing the vocab file path.

    Returns:
        Any: Parsed vocabulary mapping.
    """
    # Retrieve vocab file path from model
    vocab_path = llm.get_path_to_vocab_file()

    # Load and parse JSON vocabulary
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    return vocab
