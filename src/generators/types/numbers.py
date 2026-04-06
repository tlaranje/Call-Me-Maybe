from typing import TYPE_CHECKING
from src.utils import load_vocab
import math
import re

if TYPE_CHECKING:
    from llm_sdk import Small_LLM_Model as LLM_Model


class Numbers:
    """
    Constrained numeric generator using token-by-token validation.

    This class forces the LLM to generate only tokens that keep the
    partial string a valid prefix of a floating‑point number. It supports
    scientific notation, signs, and decimal points. The final value is
    converted to either `int` or `float` depending on configuration.
    """

    def __init__(self, type: str = "int") -> None:
        """
        Initialize the generator.

        Args:
            type (str): Desired output type. Must be `"int"` or `"float"`.
        """
        self.type = type

    def extract_number_from_prompt(self, ins: str) -> float | None:
        """
        Extract the first numeric literal from the prompt.

        Supports:
        - signed integers
        - floats
        - scientific notation (e.g., -2e-10)

        Args:
            ins (str): The input prompt.

        Returns:
            float | None: The extracted number, or None if no number exists.
        """
        NUM_PATTERN = (
            r"[+-]?"            # optional sign
            r"\d+"              # integer part
            r"(\.\d+)?"         # optional decimal part
            r"([eE][+-]?\d+)?"  # optional exponent with optional sign
        )
        m = re.search(NUM_PATTERN, ins)
        return float(m.group(0)) if m else None

    def _is_valid_extension(self, curr: str, tok: str) -> bool:
        """
        Check whether adding `tok` to `curr` keeps a valid float prefix.

        This prevents the LLM from generating tokens that would make the
        string impossible to parse as a float, except in cases where the
        prefix is still syntactically valid (e.g., "-", "2e", "2e-").

        Args:
            curr (str): Current partial numeric string.
            tok (str): Candidate token.

        Returns:
            bool: True if the token can be accepted.
        """
        s = curr + tok

        # Allow initial sign
        if s in ("-", "+"):
            return True

        # Allow sign after exponent markers
        if curr.lower().endswith(("e", "e-", "e+")) and tok in "+-":
            return True

        # Allow decimal point only once and only before exponent
        if tok == ".":
            return "e" not in curr.lower() and "." not in curr

        # Allow exponent only once and only after digits
        if tok.lower() == "e":
            return "e" not in curr.lower() and any(c.isdigit() for c in curr)

        # General case: try parsing as float
        try:
            float(s)
            return True
        except ValueError:
            return False

    def generate(self, llm: LLM_Model, ins: str) -> int | float:
        """
        Generate a number using constrained decoding.

        The LLM is queried token-by-token. Each token is accepted only if it
        keeps the partial string a valid numeric prefix. The sign is inferred
        from the number present in the original prompt.

        Args:
            llm (LLM_Model): The language model used for token generation.
            ins (str): The instruction prompt.

        Returns:
            int | float: The generated numeric value.
        """
        # Determine the sign from the prompt's original number
        prompt_value = self.extract_number_from_prompt(ins)
        sign = -1 if (prompt_value is not None and prompt_value < 0) else 1

        curr_value = ""
        curr_token = ""

        # Initial logits
        input_ids = llm.encode(ins).tolist()[0]
        logits = llm.get_logits_from_input_ids(input_ids)
        vocab = load_vocab(llm)

        # Constrained decoding loop
        while curr_token != "Ċ":
            # Greedy token selection
            curr_token = max(vocab, key=lambda s: logits[vocab[s]])

            # Accept token if it keeps the number valid
            if self._is_valid_extension(curr_value, curr_token):
                curr_value += curr_token
                input_ids = llm.encode(ins + curr_value).tolist()[0]
                logits = llm.get_logits_from_input_ids(input_ids)
            else:
                # Permanently ban invalid token
                logits[vocab[curr_token]] = -math.inf

        # Convert final string to number
        s = curr_value.lower()

        if self.type == "int":
            if "e" in s:
                base, exp = s.split("e")
                return sign * (int(float(base)) * (10 ** int(exp)))
            return sign * int(float(s))

        return sign * float(s)
