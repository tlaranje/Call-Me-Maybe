from llm_sdk import Small_LLM_Model as LLM_Model
from rich.markup import escape
from typing import Any, cast
from rich.panel import Panel
from rich.live import Live
from time import sleep
import json


def animate_field(
    live: Live,
    prompt: str,
    func_name: str,
    field: str,
    new_text: str,
    animate: bool = True
) -> None:
    """
    Animates a specific field (function name or parameters) letter by letter.

    Args:
        live (Live): The Rich Live instance for UI updates.
        prompt (str): The input prompt text.
        func_name (str): Current function name to display.
        field (str): Target field to animate ("func" or "params").
        new_text (str): The final text to be displayed/animated.
        animate (bool): Whether to perform the typing animation.
    """
    if not animate:
        # Update UI instantly without animation
        params = new_text if field != "func" else None
        func = new_text if field == "func" else func_name
        live.update(render_panel(prompt, func, params))
        return

    curr = ""
    for char in new_text:
        curr += char
        if field == "func":
            live.update(render_panel(prompt, curr, None))
        else:
            live.update(render_panel(prompt, func_name, curr))
        sleep(0.05)


def render_panel(
    prompt: str,
    func_name: str,
    params: str | None = None
) -> Panel:
    """
    Renders a Rich Panel with prompt, function, and parameter details.

    Args:
        prompt (str): The user prompt.
        func_name (str): The detected function name.
        params (str | None): JSON string of parameters or status message.

    Returns:
        Panel: A formatted Rich Panel object.
    """
    assert func_name is not None
    safe_prompt = escape(prompt)
    safe_func = escape(func_name)

    # Avoid escaping Rich tags if parameters contain error formatting
    if params and "[bold red]" in params:
        display_params = params
    else:
        display_params = escape(params) if params else "..."

    content = (
        f'[bold yellow]Prompt:[/bold yellow] {safe_prompt}\n'
        f'[bold green]Function:[/bold green] {safe_func}\n'
        f'[bold cyan]Parameters:[/bold cyan] {display_params}'
    )

    return Panel.fit(
        content,
        title="[bold white]Result[/bold white]",
        width=200
    )


def load_vocab(llm: LLM_Model) -> dict[str, int]:
    """
    Loads the vocabulary associated with the given LLM model.

    Args:
        llm (LLM_Model): The language model instance.

    Returns:
        dict[str, int]: Parsed vocabulary mapping tokens to IDs.
    """
    vocab_path = llm.get_path_to_vocab_file()

    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return cast(dict[str, int], data)


def load_json(path: str) -> Any:
    """
    Loads and parses a JSON file from the given path.

    Args:
        path (str): Path to the JSON file.

    Returns:
        Any: Parsed JSON content.

    Raises:
        FileNotFoundError: If the file does not exist at the specified path.
        ValueError: If the file content is not a valid JSON format.
    """
    try:
        with open(path, "r", encoding="utf-8") as fd:
            return json.load(fd)

    except FileNotFoundError:
        # Re-raise with a more descriptive message
        raise FileNotFoundError(f"File not found: {path}")

    except json.JSONDecodeError as e:
        # Wrap the decode error to provide file context
        raise ValueError(f"Invalid JSON format in '{path}': {e}")
