from typing import TYPE_CHECKING, Any
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

    def check_sign(self, ins: str) -> float:
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
        assert m is not None
        return -1 if m.group(0).startswith('-') else 1

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

    def generate(self, llm: LLM_Model, ins: str) -> int | float | str | Any:
        """
        Generates a numeric value based on LLM logits and regex constraints.

        Args:
            llm: The language model instance.
            ins: The input instruction string.

        Returns:
            The parsed numeric value (int or float) or None if invalid.
        """
        regex: str = '-0123456789.e\n'
        is_sci_notation: bool = False
        curr_value: Any = ""
        curr_token: str = ""
        sign: int = 1

        # Initial encoding and logit extraction
        input_ids: list[int] = llm.encode(ins + curr_value).tolist()[0]
        logits: Any = llm.get_logits_from_input_ids(input_ids)
        vocab: dict[str, int] = load_vocab(llm)

        while curr_token != 'Ċ':
            # Get the token with the highest probability
            curr_token = max(vocab.keys(), key=lambda s: logits[vocab[s]])

            # Check if the first token indicates a negative sign
            if curr_value == "" and '-' in curr_token:
                sign = -1

            # Flag if the token is scientific notation 'e'
            if curr_token == 'e':
                is_sci_notation = True

            # Combined validation: if any of these are true, reject the token
            is_invalid = (
                (curr_token == '-' and not curr_value.endswith('e')) or
                (is_sci_notation and curr_token == '.') or
                (curr_token not in regex) or
                (curr_value.count('.') >= 2)
            )

            if is_invalid:
                logits[vocab[curr_token]] = -math.inf
                continue

            # Append valid token and refresh logits for the next step
            curr_value += curr_token
            input_ids = llm.encode(ins + curr_value).tolist()[0]
            logits = llm.get_logits_from_input_ids(input_ids)

        # Final logic for parsing the collected string into a number
        try:
            if is_sci_notation:
                return sign * float(curr_value)

            dots: int = curr_value.count('.')

            if self.type == "float":
                return sign * float(curr_value)

            if self.type == "int" and dots == 0:
                return sign * int(curr_value)

        except (ValueError, TypeError, OverflowError):
            return None

        return None
