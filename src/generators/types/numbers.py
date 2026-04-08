from typing import TYPE_CHECKING, Any
from decimal import Decimal, InvalidOperation
from src.utils import load_vocab
import math


if TYPE_CHECKING:
    from llm_sdk import Small_LLM_Model as LLM_Model


class Numbers:
    """
    Constrained numeric generator using token-by-token validation.

    This class forces the LLM to generate only tokens that keep the partial
    string a valid prefix of a numeric literal (int, float, or scientific).
    """

    def __init__(self, type_str: str = "int") -> None:
        """
        Initializes the generator with a target numeric type.

        Args:
            type_str (str): Desired output type, either "int" or "float".
        """
        self.type = type_str

    def _is_valid_extension(self, curr: str, tok: str) -> bool:
        """
        Checks if adding a token maintains a valid numeric prefix.

        Args:
            curr (str): Current accumulated numeric string.
            tok (str): Candidate token to append.

        Returns:
            bool: True if the resulting string is a valid numeric prefix.
        """
        s = curr + tok

        if s in ("-", "+"):
            return True

        if curr.lower().endswith(("e", "e-", "e+")) and tok in "+-":
            return True

        if tok == ".":
            return "e" not in curr.lower() and "." not in curr

        if tok.lower() == "e":
            return "e" not in curr.lower() and any(c.isdigit() for c in curr)

        try:
            float(s)
            return True
        except ValueError:
            return False

    def generate(self, llm: "LLM_Model", ins: str) -> Any:
        """
        Generates a numeric value based on LLM logits and constraints.

        Args:
            llm (LLM_Model): The language model instance.
            ins (str): The input instruction prompt.

        Returns:
            Any: The parsed numeric value or None if invalid.
        """
        allowed_chars: str = '-0123456789.e\n'
        is_sci_notation: bool = False
        curr_value: str = ""
        curr_token: str = ""

        vocab = load_vocab(llm)

        while curr_token != 'Ċ':
            # Encode current state to get fresh logits
            input_ids = llm.encode(ins + curr_value).tolist()[0]
            logits = llm.get_logits_from_input_ids(input_ids)

            while True:
                # Greedy selection: pick the token with highest probability
                curr_token = max(vocab.keys(), key=lambda s: logits[vocab[s]])
                if curr_token == 'Ċ':
                    break

                if curr_token == 'e':
                    is_sci_notation = True

                # Validation logic
                is_minus_valid = (
                    curr_token == '-' and
                    (curr_value == "" or curr_value.lower().endswith('e'))
                )

                is_invalid = (
                    (curr_token == '-' and not is_minus_valid) or
                    (is_sci_notation and curr_token == '.') or
                    (curr_token not in allowed_chars)
                )

                if is_invalid:
                    # Penalize invalid token and try the next best
                    logits[vocab[curr_token]] = -math.inf
                    continue

                break

            if curr_token == 'Ċ':
                break

            curr_value += curr_token

        return self._parse_result(curr_value, is_sci_notation)

    def _parse_result(self, v_str: str, is_sci: bool) -> Any:
        """
        Parses the generated string into the final number.

        Args:
            v_str (str): The raw string from the LLM.
            is_sci (bool): Whether scientific notation was detected.

        Returns:
            Union[int, float, None]: The final number or None.
        """
        try:
            if not v_str:
                return None

            if is_sci or self.type == "float":
                result = float(v_str)

                # Precision check for zero
                if result == 0.0 and Decimal(v_str) != 0:
                    return None

                if math.isinf(result) or math.isnan(result):
                    return None

                return result

            if self.type == "int" and "." not in v_str:
                return int(v_str)

        except (ValueError, TypeError, OverflowError, InvalidOperation):
            return None

        return None
