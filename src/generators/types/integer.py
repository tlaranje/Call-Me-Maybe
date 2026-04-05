from typing import TYPE_CHECKING
from src.utils import load_vocab
import math

if TYPE_CHECKING:
    from llm_sdk import Small_LLM_Model as LLM_Model


class Integer:
    def _is_valid_prefix(self, s: str) -> bool:
        """Check if string is a valid numeric prefix (e.g. '-', '-1', '1.')"""
        if not s:
            return True
        try:
            float(s)
            return True
        except ValueError:
            # Valid prefixes that aren't yet parsable
            return s in {'-', '.'} or s.startswith('-') and s[1:] in {'', '.'}

    def generate(self, llm: LLM_Model, ins: str) -> int:
        """
        Generate a numeric parameter using constrained decoding.

        Builds a number token-by-token, only accepting tokens that
        keep the string parsable as a valid finite float.

        Args:
            llm (LLM_Model): Language model used for generation.
            ins (str): Instruction prompt.
            p_name (str): Parameter name.

        Returns:
            dict[str, float]: Generated numeric value mapped to parameter name.
        """
        curr_value = ""
        curr_token = ""

        # Initial encoding
        input_ids = llm.encode(ins + curr_value).tolist()[0]
        logits = llm.get_logits_from_input_ids(input_ids)
        vocab = load_vocab(llm)

        while curr_token != 'Ċ':
            # Select highest probability token (greedy decoding)
            curr_token = max(vocab.keys(), key=lambda s: logits[vocab[s]])

            try:
                # Check if token keeps valid float
                s_value = curr_value + curr_token
                value = float(s_value)

                if math.isfinite(value):
                    # Accept token and update context
                    curr_value += curr_token
                    input_ids = llm.encode(ins + curr_value).tolist()[0]
                    logits = llm.get_logits_from_input_ids(input_ids)
                else:
                    if self._is_valid_prefix(s_value):
                        curr_value = s_value
                        input_ids = llm.encode(ins + curr_value).tolist()[0]
                        logits = llm.get_logits_from_input_ids(input_ids)
                    else:
                        logits[vocab[curr_token]] = -math.inf

            except ValueError:
                # Reject invalid float sequences
                logits[vocab[curr_token]] = -math.inf

        return int(float(curr_value))
