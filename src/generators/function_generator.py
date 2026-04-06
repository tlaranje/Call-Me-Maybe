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
            "Only choose the best function based on the user prompt "
            "if you can do the prompt wiht the function you choose.\n"
            "Return ONLY the function name.\n"
            "IMPORTANT RULES:\n"
            "- 'negative', 'positive', 'prime', 'factorial', "
            "'absolute' → fn_none\n"
            "- fn_is_even ONLY for 'even' or 'odd' questions\n"
            # "- If no function matches the prompt, return fn_none\n"
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

    FUNCTION_WHITELIST = {
        "fn_is_even": ["even", "odd"],
        "fn_get_square_root": ["square root", "sqrt", "√"],
        "fn_add_numbers": ["sum", "add", "plus", "addition", "total"],
        "fn_reverse_string": ["reverse", "flip", "backwards"],
        "fn_greet": ["greet", "hello", "hi", "hey"],
        "fn_filter_list": ["filter", "list", "sort by", "order by"],
    }

    FUNCTION_BLACKLIST: dict = {}

    def is_valid_match(self, func_name: str, prompt: str) -> bool:
        prompt_lower = prompt.lower()

        # Whitelist: função só é válida se prompt contém pelo menos uma palavra
        whitelist = self.FUNCTION_WHITELIST.get(func_name, [])
        if whitelist and not any(word in prompt_lower for word in whitelist):
            return False

        # Blacklist: função é inválida se prompt contém palavra bloqueada
        blacklist = self.FUNCTION_BLACKLIST.get(func_name, [])
        if any(word in prompt_lower for word in blacklist):
            return False

        return True

    def generate(
        self,
        llm: LLM_Model,
        funcs: list[FunctionDefinition],
        prompt: str,
        excluded=None,
        max_retries=3
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
        if excluded is None:
            excluded = set()
        if len(excluded) >= max_retries:
            return ""
        # List of valid function names (sorted for deterministic behavior)
        filtered_funcs = [f for f in funcs if f.name not in excluded]

        f_names = sorted([f.name for f in filtered_funcs])
        # f_names_with_none = sorted(f_names + ["fn_none"])
        f_names_with_none = sorted(f_names)

        # All function names start with this prefix
        output = "fn_"

        # Build the instruction prompt for the LLM
        instructions = self.get_func_instructions(filtered_funcs, prompt)

        # Vocabulary: token string → token ID
        vocab: Any = load_vocab(llm)

        def encode_and_get_logits() -> tuple[list[int], list[float]]:
            """Encode current text and get logits for the next token."""
            ids = llm.encode(instructions + output).tolist()[0]
            return ids, llm.get_logits_from_input_ids(ids)

        # Logits for the initial prefix "fn_"
        input_ids, logits = encode_and_get_logits()

        # --- Main generation loop -----------------------------------------
        # Continue until the output exactly matches a function name.
        while output not in f_names_with_none:
            # Keep only names that still match the current prefix
            ft_list = [f for f in f_names_with_none if f.startswith(output)]

            # Only one possible function → auto-complete
            if len(ft_list) == 1:
                output += ft_list[0][len(output):]
                break

            # No possible function → invalid path
            if not ft_list:
                return ""

            # Collect tokens that keep the prefix valid
            valid_tokens = []
            for s, tid in vocab.items():
                token_str = s[1:] if s.startswith("Ġ") else s

                # Valid if it keeps the prefix matching some function
                if token_str and any(
                    f.startswith(output + token_str) for f in f_names_with_none
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
        if output == "fn_none":
            return ""
        # Se não é válido, retry excluindo esta função
        if not self.is_valid_match(output, prompt):
            return self.generate(llm, funcs, prompt, excluded | {output})
        return "" if output == "fn_none" else output
