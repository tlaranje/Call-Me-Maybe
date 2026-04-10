import re
from typing import TYPE_CHECKING, Any, Optional

from src.utils import load_vocab

if TYPE_CHECKING:
    from llm_sdk import Small_LLM_Model as LLM_Model


class String:
    """
    Generator for string-type parameters using the LLM.

    This class provides a mechanism to generate string values token-by-token
    by interacting with a Large Language Model until specific termination
    criteria are met.

    Attributes:
        _LEAK_PATTERN (re.Pattern): Compiled regex to detect if the LLM
            starts generating subsequent parameters (e.g., ' key=').
    """

    _LEAK_PATTERN = re.compile(r"\s+\w+\s*=|['\"],\s*['\"]?\w")

    def generate(
        self, llm: "LLM_Model", ins: str,
    ) -> Any:
        """
        Generates a string value token by token until a stop sequence is met.

        Args:
            llm (LLM_Model): The language model instance.
            ins (str): The instruction prompt leading up to the value.

        Returns:
            Any: The generated string value, stripped of surrounding whitespace
                and quotes.
        """
        curr_value = ""
        vocab = load_vocab(llm)
        max_tokens = 100
        iterations = 0
        stop_tokens = {"Ċ", "\n"}

        # Identify if the string starts with a quote to handle closure
        stripped_ins = ins.rstrip()
        quote_char: Optional[str] = (
            stripped_ins[-1]
            if stripped_ins and stripped_ins[-1] in ('"', "'")
            else None
        )

        while iterations < max_tokens:
            # Encode current state and get logits
            input_ids = llm.encode(ins + curr_value).tolist()[0]
            logits = llm.get_logits_from_input_ids(input_ids)

            # Select the token with the highest probability (greedy decoding)
            raw_token = max(
                vocab.keys(),
                key=lambda s: logits[vocab[s]],
            )
            # Handle special BPE whitespace character
            processed_token = raw_token.replace("Ġ", " ")

            # 1. Stop on newline tokens
            if any(stop in raw_token for stop in stop_tokens):
                clean = processed_token
                for stop in stop_tokens:
                    clean = clean.split(stop)[0]
                curr_value += clean
                break

            # 2. Stop at closing quote if a matching start quote was found
            if quote_char and quote_char in processed_token:
                curr_value += processed_token.split(quote_char)[0]
                break

            curr_value += processed_token

            # 3. Stop if leaked into next parameter (regex check)
            match = self._LEAK_PATTERN.search(curr_value)
            if match:
                curr_value = curr_value[: match.start()]
                break

            iterations += 1

        # Final string cleaning and quote removal
        result = curr_value.strip()
        if (
            quote_char
            and result.startswith(quote_char)
            and result.endswith(quote_char)
        ):
            result = result[1:-1]

        return result.strip()
