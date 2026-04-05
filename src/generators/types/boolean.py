from typing import TYPE_CHECKING
from src.utils import load_vocab
import math

if TYPE_CHECKING:
    from llm_sdk import Small_LLM_Model as LLM_Model


class Boolean:
    def generate(self, llm: LLM_Model, ins: str) -> bool:
        BOOL_TOKENS = {"true", "false"}

        input_ids = llm.encode(ins).tolist()[0]
        logits = llm.get_logits_from_input_ids(input_ids)
        vocab = load_vocab(llm)

        # Mask: keep only tokens that start a valid bool ("true" / "false")
        masked_logits = {k: -math.inf for k in vocab}
        for token, idx in vocab.items():
            clean = token.lstrip('Ġ').lower()
            if any(b.startswith(clean) for b in BOOL_TOKENS):
                masked_logits[token] = logits[idx]

        curr_value = ""

        while True:
            raw_token = max(
                masked_logits.keys(), key=lambda s: masked_logits[s]
            )
            clean_token = raw_token.lstrip('Ġ').lower()

            curr_value += clean_token

            # Check if we already have a complete bool
            if curr_value in BOOL_TOKENS:
                break

            # Check if curr_value is still a valid prefix of any bool
            valid_prefixes = [
                b for b in BOOL_TOKENS if b.startswith(curr_value)
            ]
            if not valid_prefixes:
                # Backtrack: invalid path, suppress token and retry
                masked_logits[raw_token] = -math.inf
                curr_value = curr_value[: -len(clean_token)]
                continue

            # Re-run logits for next token
            input_ids = llm.encode(ins + curr_value).tolist()[0]
            logits = llm.get_logits_from_input_ids(input_ids)

            # Re-mask: only tokens that extend a valid bool prefix
            masked_logits = {k: -math.inf for k in vocab}
            for token, idx in vocab.items():
                clean = token.lstrip('Ġ').lower()
                candidate = curr_value + clean
                if any(b.startswith(candidate) for b in BOOL_TOKENS):
                    masked_logits[token] = logits[idx]

        return curr_value == "true"
