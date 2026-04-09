from typing import TYPE_CHECKING
from src.utils import load_vocab

if TYPE_CHECKING:
    from llm_sdk import Small_LLM_Model as LLM_Model


class String:
    """
    Generator for string-type parameters using the LLM.
    """

    def generate(self, llm: "LLM_Model", ins: str, prompt: str = "") -> str:
        """
        Generates a string value token by token until a stop sequence is met.

        Args:
            llm (LLM_Model): The language model instance.
            ins (str): The instruction prompt leading up to the value.

        Returns:
            str: The generated string value.
        """
        curr_value = ""
        vocab = load_vocab(llm)
        max_tokens = 100
        iterations = 0

        # Characters that signal the end of the string value
        stop_tokens = {'Ċ'}

        while iterations < max_tokens:
            # Encode current context and get model predictions
            input_ids = llm.encode(ins + curr_value).tolist()[0]
            logits = llm.get_logits_from_input_ids(input_ids)

            # Greedy selection: pick the token with the highest logit
            raw_token = max(vocab.keys(), key=lambda s: logits[vocab[s]])

            # Convert BPE artifacts (like 'Ġ') to readable spaces
            processed_token = raw_token.replace('Ġ', ' ')

            # Check if the model generated a stop token (like a newline)
            if any(stop in raw_token for stop in stop_tokens):
                # Cut the string at the first stop character found
                clean_token = processed_token.split('\n')[0].split('Ċ')[0]
                curr_value += clean_token
                break

            curr_value += processed_token
            iterations += 1

        return curr_value.strip()
