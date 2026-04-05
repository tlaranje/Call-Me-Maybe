from typing import TYPE_CHECKING
from src.utils import load_vocab
import math

if TYPE_CHECKING:
    from llm_sdk import Small_LLM_Model as LLM_Model


class String:
    def generate(self, llm: LLM_Model, ins: str) -> str:
        """
        Generate a string parameter using constrained decoding.

        Handles BPE tokens (e.g., 'Ġ' for spaces) and reconstructs
        a natural string until a newline token is reached.

        Args:
            llm (LLM_Model): Language model used for generation.
            ins (str): Instruction prompt.
            p_name (str): Parameter name.

        Returns:
            dict[str, str]: Generated string mapped to parameter name.
        """
        curr_value = ""
        curr_token = ""

        input_ids = llm.encode(ins + curr_value).tolist()[0]
        logits = llm.get_logits_from_input_ids(input_ids)
        vocab = load_vocab(llm)

        STOP_TOKENS = {'Ċ', ',', '"'}
        while curr_token not in STOP_TOKENS:
            # Greedy token selection
            raw_token = max(vocab.keys(), key=lambda s: logits[vocab[s]])

            # Handle BPE space prefix
            if raw_token.startswith('Ġ') and curr_value:
                curr_token = ' ' + raw_token[1:]
            else:
                curr_token = (
                    raw_token[1:] if raw_token.startswith('Ġ') else raw_token
                )

            # Skip empty tokens
            if not curr_token:
                logits[vocab[raw_token]] = -math.inf
                continue

            # Stop condition (newline token)
            if 'Ċ' in curr_token:
                curr_value += curr_token.split('Ċ')[0]
                break

            # Append token and continue decoding
            if curr_token != ',':
                curr_value += curr_token
            input_ids = llm.encode(ins + curr_value).tolist()[0]
            logits = llm.get_logits_from_input_ids(input_ids)

            # Fix escaped characters
            curr_value = curr_value.replace('\\\\', '\\')
            curr_value = curr_value.strip("'\"")
        return curr_value
