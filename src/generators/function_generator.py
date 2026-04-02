from src.models import FunctionDefinition
from llm_sdk import Small_LLM_Model as LLM_Model
from src.utils import render_panel, load_vocab
from rich.live import Live
from typing import Any
import time


class FunctionGenerator:
    def get_func_instructions(
        self, funcs: list[FunctionDefinition], prompt: str
    ) -> str:
        """
        Create instructions for the LLM to select the best function.

        Builds a structured prompt containing system rules, available
        functions, and the user input.

        Args:
            funcs (list[FuncDef]): Available function definitions.
            prompt (str): User input in natural language.

        Returns:
            str: Instruction string formatted for the LLM.
        """
        return (
            "<|im_start|>system\n"
            "You are a function selector.\n"
            "Choose the best function based on the user prompt.\n"
            "Return ONLY the function name.\n"
            "IMPORTANT RULES:\n"
            "- Strings must not have newline in the midle\n"
            "- 'Calculate compound' means (principal * (1 + rate)^years) "
            "but not means product\n"
            "- 'reverse' means flipping text order\n"
            "- 'replace', 'substitute', 'change' "
            "means modifying parts of text\n"
            "- 'numbers', 'vowels', 'words' → use substitution function\n\n"
            + str(funcs) + "\n<|im_end|>\n"
            "<|im_start|>user\n" + prompt + "\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def generate_function_name(
        self,
        llm: LLM_Model,
        funcs: list[FunctionDefinition],
        prompt: str,
        live: Live | None = None
    ) -> str:
        """
        Generate the most appropriate function name for a given prompt.

        Uses constrained token-by-token generation to ensure the output
        matches one of the available function names. Optionally streams
        progress to a Rich Live panel.

        Args:
            llm (LLM_Model): Language model used for generation.
            funcs (list[FuncDef]): List of available functions.
            prompt (str): User input prompt.
            live (optional): Rich Live instance for streaming UI updates.

        Returns:
            str: Selected function name.
            Returns empty string if no match is found.
        """
        # Extract valid function names
        f_names = sorted([f.name for f in funcs])

        # Prefix expected by the model
        output = "fn_"

        # Build LLM instructions
        instructions = self.get_func_instructions(funcs, prompt)

        # Load token vocabulary
        vocab: Any = load_vocab(llm)

        def render() -> None:
            """Update UI with current partial output."""
            if live:
                live.update(render_panel(prompt, output, None))

        def encode_and_get_logits() -> tuple[int, list[float]]:
            """
            Encode current sequence and retrieve logits
            for next-token prediction.
            """
            ids = llm.encode(instructions + output).tolist()[0]
            return ids, llm.get_logits_from_input_ids(ids)

        # Initial render (fn_)
        render()

        input_ids, logits = encode_and_get_logits()

        while output not in f_names:
            # Filter functions matching current prefix
            ft_list = [f for f in f_names if f.startswith(output)]

            # If only one match remains → autocomplete
            if len(ft_list) == 1:
                for char in ft_list[0][len(output):]:
                    output += char
                    render()
                    time.sleep(0.05)
                break

            # No valid continuation → fail
            if not ft_list:
                return ""

            # Collect valid next tokens
            valid_tokens = []
            for s, tid in vocab.items():
                token_str = s[1:] if s.startswith("Ġ") else s

                if token_str and any(
                    f.startswith(output + token_str) for f in f_names
                ):
                    valid_tokens.append((tid, token_str))

            if not valid_tokens:
                return ""

            # Select best token based on logits
            token_id, token_str = max(
                valid_tokens, key=lambda x: logits[x[0]]
            )

            # Append token character by character (typing effect)
            for char in token_str:
                output += char
                render()
                time.sleep(0.05)

            # Recompute logits for new sequence
            input_ids, logits = encode_and_get_logits()

        return output
