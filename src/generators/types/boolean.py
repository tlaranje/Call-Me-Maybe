from src.utils import load_vocab
from typing import TYPE_CHECKING
import math


if TYPE_CHECKING:
    from llm_sdk import Small_LLM_Model as LLM_Model


class Boolean:
    """
    Generator for boolean-type parameters using constrained token selection.
    """

    def generate(self, llm: "LLM_Model", ins: str) -> bool:
        """
        Generates a boolean value by masking logits for 'true' and 'false'.

        Args:
            llm (LLM_Model): The language model instance.
            ins (str): The instruction prompt leading up to the value.

        Returns:
            bool: True if the model generated 'true', False otherwise.
        """
        bool_tokens = {"true", "false"}
        vocab = load_vocab(llm)

        # Initial logit extraction
        input_ids = llm.encode(ins).tolist()[0]
        logits = llm.get_logits_from_input_ids(input_ids)

        # Initial mask: only keep tokens that could start "true" or "false"
        masked_logits = {k: -math.inf for k in vocab}
        for token, idx in vocab.items():
            clean = token.lstrip('Ġ').lower()
            if any(b.startswith(clean) for b in bool_tokens):
                masked_logits[token] = logits[idx]

        curr_value = ""

        while True:
            # Pick the best token from the allowed candidates
            raw_token = max(
                masked_logits.keys(), key=lambda s: masked_logits[s]
            )
            clean_token = raw_token.lstrip('Ġ').lower()
            curr_value += clean_token

            # Stop if we have reached a full boolean string
            if curr_value in bool_tokens:
                break

            # Check if current progress can still form a valid boolean
            valid_prefixes = [
                b for b in bool_tokens if b.startswith(curr_value)
            ]

            if not valid_prefixes:
                # Backtrack: the chosen token leads to a dead end
                masked_logits[raw_token] = -math.inf
                curr_value = curr_value[:-len(clean_token)]
                continue

            # Refresh logits for the next token based on new context
            input_ids = llm.encode(ins + curr_value).tolist()[0]
            logits = llm.get_logits_from_input_ids(input_ids)

            # Re-mask: allow only tokens that extend the current valid prefix
            masked_logits = {k: -math.inf for k in vocab}
            for token, idx in vocab.items():
                clean = token.lstrip('Ġ').lower()
                candidate = curr_value + clean
                if any(b.startswith(candidate) for b in bool_tokens):
                    masked_logits[token] = logits[idx]

        return curr_value == "true"
