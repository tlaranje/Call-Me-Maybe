from src.models import FunctionDefinition
from llm_sdk import Small_LLM_Model as LLM_Model
from src.utils import load_vocab
from typing import Any


class FunctionGenerator:
    def get_func_instructions(
        self, funcs: list[FunctionDefinition], prompt: str
    ) -> str:
        """
        Create instructions for the LLM to select the best function.

        Builds a structured prompt containing system rules, available
        functions, and the user input.

        Args:
            funcs (list[FuncDef]): Available function definitions.
            prompt (str): User input in natural language.

        Returns:
            str: Instruction string formatted for the LLM.
        """
        return (
            "<|im_start|>system\n"
            "You are a function selector.\n"
            "Choose the best function based on the user prompt.\n"
            "Return ONLY the function name.\n"
            "IMPORTANT RULES:\n"
            "- Strings must not have newline in the midle\n"
            "- 'Calculate compound' means (principal * (1 + rate)^years) "
            "but not means product\n"
            "- 'reverse' means flipping text order\n"
            "- 'replace', 'substitute', 'change' "
            "means modifying parts of text\n"
            "- 'numbers', 'vowels', 'words' → use substitution function\n\n"
            + str(funcs) + "\n<|im_end|>\n"
            "<|im_start|>user\n" + prompt + "\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def generate(
        self,
        llm: LLM_Model,
        funcs: list[FunctionDefinition],
        prompt: str
    ) -> str:
        """
        Generate the most appropriate function name for a given prompt.

        This method performs constrained, token-by-token generation to
        ensure the output always corresponds to one of the available
        function names. The model is guided through a restricted decoding
        process that only allows continuations matching valid function-name
        prefixes.

        Args:
            llm (LLM_Model):
                The language model responsible for next token prediction.
            funcs (list[FunctionDefinition]):
                The list of available function definitions from which the
                model must select.
            prompt (str):
                The natural‑language user request that determines which
                function should be chosen.

        Returns:
            str:
                The selected function name. Returns an empty string if the
                model cannot produce a valid continuation or confidence is
                insufficient.
        """
        # List of valid function names (sorted for deterministic behavior)
        f_names = sorted([f.name for f in funcs])

        # All function names start with this prefix
        output = "fn_"

        # Build the instruction prompt for the LLM
        instructions = self.get_func_instructions(funcs, prompt)

        # Vocabulary: token string → token ID
        vocab: Any = load_vocab(llm)

        def encode_and_get_logits() -> tuple[list[int], list[float]]:
            """Encode current text and get logits for the next token."""
            ids = llm.encode(instructions + output).tolist()[0]
            return ids, llm.get_logits_from_input_ids(ids)

        # Logits for the initial prefix "fn_"
        input_ids, logits = encode_and_get_logits()

        # --- Initial confidence check -------------------------------------
        # Check if the model is confident about ANY valid next character.
        # If not, the prompt likely doesn't match any function.
        initial_scores = []
        for f in f_names:
            next_char = f[len("fn_")]  # first letter after "fn_"
            for s, tid in vocab.items():
                token_str = s[1:] if s.startswith("Ġ") else s
                if token_str == next_char:
                    initial_scores.append(logits[tid])

        if not initial_scores or max(initial_scores) < 5:
            return ""

        # --- Main generation loop -----------------------------------------
        # Continue until the output exactly matches a function name.
        while output not in f_names:

            # Keep only names that still match the current prefix
            ft_list = [f for f in f_names if f.startswith(output)]

            # Only one possible function → auto-complete
            if len(ft_list) == 1:
                output += ft_list[0][len(output):]
                return output

            # No possible function → invalid path
            if not ft_list:
                return ""

            # Collect tokens that keep the prefix valid
            valid_tokens = []
            for s, tid in vocab.items():
                token_str = s[1:] if s.startswith("Ġ") else s

                # Valid if it keeps the prefix matching some function
                if token_str and any(
                    f.startswith(output + token_str) for f in f_names
                ):
                    valid_tokens.append((tid, token_str))

            # No valid continuation
            if not valid_tokens:
                return ""

            # Choose the token with the highest logit among valid options
            token_id, token_str = max(
                valid_tokens, key=lambda x: logits[x[0]]
            )

            # Append token to the output
            output += token_str

            # Recompute logits for the updated sequence
            input_ids, logits = encode_and_get_logits()

        # Completed valid function name
        return output
