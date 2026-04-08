from typing import Any, Optional, TYPE_CHECKING
from src.utils import load_vocab
import numpy as np


if TYPE_CHECKING:
    from src.models import FunctionDefinition


class FunctionGenerator:
    """
    Handles the logical selection of functions based on LLM logits and rules.
    """

    def get_func_instructions(
        self, funcs: list[FunctionDefinition], prompt: str
    ) -> str:
        """
        Constructs the system prompt for function selection.

        Args:
            funcs (list): List of available FunctionDefinition objects.
            prompt (str): The user input prompt.

        Returns:
            str: The formatted instructions for the LLM.
        """
        func_descriptions = "\n".join(
            [f"- {f.name}: {f.description}" for f in funcs]
        )
        return (
            "<|im_start|>system\n"
            "You are a precise function selector.\n"
            "STRICT RULES:\n"
            "1. For math (add, multiply, root), use 'fn_add_numbers' "
            "or 'fn_get_square_root'.\n"
            "2. For pattern replacement (vowels, numbers, regex), "
            "ALWAYS use 'fn_substitute_string_with_regex'.\n"
            "3. Use 'fn_format_template' ONLY for filling "
            "'{name}' variables.\n"
            "4. If no match exists, return 'fn_none'.\n\n"
            f"Available Functions:\n{func_descriptions}\n"
            "Selection: <|im_end|>\n"
            f"<|im_start|>user\n{prompt}\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def it_match(self, func_name: str, prompt: str) -> bool:
        """
        Validates if the chosen function makes sense for the given prompt.

        Args:
            func_name (str): Name of the selected function.
            prompt (str): The user input prompt.

        Returns:
            bool: True if it is a semantic match, False otherwise.
        """
        p = prompt.lower()

        # Rules for SQL Execution
        if func_name == "fn_execute_sql_query":
            sql_must_have = [
                "select", "update", "delete", "insert",
                "table", "database", "query"
            ]
            # Ensure the prompt actually mentions database-related actions
            if not any(s in p for s in sql_must_have):
                return False

            # Block if it looks like a file path or basic math to avoid false
            # positives where SQL keywords might appear out of context
            if any(x in p for x in ["/", ".", "add", "sum"]):
                return False

        # Rules for Template Formatting
        if func_name == "fn_format_template":
            # If the user asks for pattern replacement, it's a job for Regex,
            # not for simple template formatting
            regex_keywords = [
                "all numbers", "all vowels", "asterisks", "replace all"
            ]
            if any(k in p for k in regex_keywords):
                return False

            # A template function requires explicit place-holders or
            # the word 'template' to be considered a valid match
            if "{" not in p and "template" not in p:
                return False

        # Rules for Regex Substitution
        if func_name == "fn_substitute_string_with_regex":
            # Guardrail: prevent regex from intercepting mathematical
            # operations that might contain similar sounding keywords
            if any(m in p for m in ["sum of", "plus", "root"]):
                return False

        return True

    def get_probability(
            self, logits: Any, valid_token_ids: list[int]
    ) -> float:
        """
        Calculates softmax probability for valid tokens.

        Args:
            logits (Any): Raw output logits from the LLM.
            valid_token_ids (list[int]): IDs of tokens that fit valid prefixes.

        Returns:
            float: The maximum probability among valid tokens.
        """
        if not valid_token_ids:
            return 0.0

        # Extract only the logits corresponding to the allowed tokens
        valid_logits = np.array([logits[tid] for tid in valid_token_ids])

        # Apply numerical stability: subtract the max value before
        # exponentiation to prevent potential overflow
        # (Standard Softmax implementation)
        exp_logits = np.exp(valid_logits - np.max(valid_logits))

        # Normalize the exponential values so they sum up to 1.0
        # (probabilities)
        probs = exp_logits / exp_logits.sum()

        # Return the highest probability found among the valid candidates
        return float(np.max(probs))

    def generate(
        self,
        llm: Any,
        funcs: list[FunctionDefinition],
        prompt: str,
        excluded: Optional[set[str]] = None
    ) -> str:
        """
        Generates a function name by validating tokens and semantic rules.

        Args:
            llm (Any): The LLM model instance.
            funcs (list): List of available functions.
            prompt (str): The user prompt.
            excluded (Optional[set]): Functions to ignore in this pass.

        Returns:
            str: The selected function name.
        """
        if excluded is None:
            excluded = set()

        # Get names of functions that haven't been excluded yet
        f_names = [f.name for f in funcs if f.name not in excluded]
        f_names_with_none = sorted(list(set(f_names) | {"fn_none"}))

        # Start every generation with the prefix 'fn_'
        output = "fn_"
        vocab = load_vocab(llm)
        instructions = self.get_func_instructions(funcs, prompt)
        threshold = 0.9

        # Build the function name token by token
        while output not in f_names_with_none:
            # List functions that start with our current text
            ft_list = [f for f in f_names_with_none if f.startswith(output)]

            if not ft_list:
                return ""

            # If only one option is left, we found our function
            if len(ft_list) == 1:
                output = ft_list[0]
                break

            # Ask the LLM for the next most likely tokens
            ids = llm.encode(instructions + output).tolist()[0]
            logits = llm.get_logits_from_input_ids(ids)

            # Only allow tokens that help complete a valid function name
            valid_tokens = []
            for s, tid in vocab.items():
                # Clean up the space character artifact ('Ġ') from the token
                t_str = s[1:] if s.startswith("Ġ") else s
                if t_str and any(
                    f.startswith(output + t_str) for f in ft_list
                ):
                    valid_tokens.append((tid, t_str))

            if not valid_tokens:
                return ""

            # Calculate how confident the model is about these choices
            prob = self.get_probability(logits, [vt[0] for vt in valid_tokens])

            # If confidence is too low, restart and try a different function
            if prob < threshold and output != "fn_":
                return self.generate(llm, funcs, prompt, excluded | {output})

            # Pick the best token from the allowed list
            _, best_tstr = max(valid_tokens, key=lambda x: logits[x[0]])
            output += best_tstr

        # Final check: does the function actually match the user's intent?
        if output not in ("", "fn_none"):
            if not self.it_match(output, prompt):
                # If it's a mismatch, exclude this function and try again
                return self.generate(llm, funcs, prompt, excluded | {output})

        return "" if output == "fn_none" else output
