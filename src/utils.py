from llm_sdk import Small_LLM_Model as LLM_Model
from rich.panel import Panel
from typing import Any
import json


def render_panel(
    prompt: str,
    func_name: str | None = None,
    params: dict | None = None
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
    return Panel.fit(
        f'[bold bright_yellow]Prompt:[/bold bright_yellow] {prompt}\n'
        f'[bold green]Function:[/bold green] {func_name or "..."}\n'
        f'[bold cyan]Parameters:[/bold cyan] {params or "..."}',
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
