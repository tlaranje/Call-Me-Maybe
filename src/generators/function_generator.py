from typing import Any, TYPE_CHECKING, List
import re

from src.utils import load_vocab

if TYPE_CHECKING:
    from src.models import FunctionDefinition


class FunctionGenerator:
    """
    Handles the selection and generation of function calls using an LLM.

    This class filters available functions based on the user prompt and
    constrains the LLM's decoding process to ensure it only generates
    valid function names.
    """

    def get_func_instructions(
        self, funcs: List["FunctionDefinition"], prompt: str
    ) -> str:
        """
        Generates the system prompt for the LLM.

        Constructs a prompt using ChatML format containing strict rules
        and the list of available functions.

        Args:
            funcs (List[FunctionDefinition]): List of functions to describe.
            prompt (str): The user's input query.

        Returns:
            str: The formatted system and user prompt.
        """
        func_descriptions = "\n".join(
            [f"- {f.name}: {f.description}" for f in funcs]
        )
        return (
            "<|im_start|>system\n"
            "You are a strict function selector.\n"
            "RULES:\n"
            "1. If the prompt contains 'replace', 'with', or 'regex', "
            "you MUST use 'fn_substitute_string_with_regex'.\n"
            "2. NEVER use 'fn_execute_sql_query' unless keywords like "
            "'SELECT', 'INSERT', 'UPDATE' or 'DATABASE' are present.\n"
            "3. If no function matches the intent exactly, return 'fn_none'.\n"
            "4. NEVER use 'fn_greet' unless the prompt explicitly asks to "
            "greet or say hello to a person.\n"
            "5. NEVER use any function if the prompt asks for something "
            "none of the functions can do.\n"
            "6. NEVER use 'fn_get_square_root' unless the prompt explicitly "
            "mentions 'square root', 'sqrt', 'square root of', or "
            "'raiz quadrada'. The word 'square' alone does NOT trigger "
            "this function.\n"
            "7. NEVER use 'fn_execute_sql_query' for mathematical "
            "expressions like '^', 'power', or 'exponent'.\n"
            "8. NEVER use 'fn_calculate_compound_interest' unless the prompt "
            "explicitly mentions 'interest', 'rate', "
            "'principal', or 'compound'.\n"
            f"Available Functions:\n{func_descriptions}\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n{prompt}\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def filter_functions(
        self, funcs: List["FunctionDefinition"], p: str
    ) -> List["FunctionDefinition"]:
        """
        Filters the function list based on keywords and parameter counts.

        Uses regex to extract entities from the prompt and compares them
        against function metadata to reduce the search space for the LLM.

        Args:
            funcs (List[FunctionDefinition]): Total available functions.
            p (str): The user prompt.

        Returns:
            List[FunctionDefinition]: A subset of functions likely to match.
        """
        # Remove alphanumerics like 'var123' and lowercase everything
        clean_p = re.sub(r'[a-z]+\d+|\d+[a-z]+', '', p.lower())
        prompt_entities = set(re.findall(r'[a-z]{3,}', clean_p))

        # Identify isolated numbers in the prompt
        numbers = re.findall(r'(?<![a-z0-9-])[-+]?\d+\.?\d*(?![a-z0-9-])', p)
        num_count = len(numbers)

        stop_words = {
            "the", "and", "with", "for", "this", "that", "from",
            "into", "all", "your", "using", "query", "what", "is"
        }
        math_types = {"number", "integer", "float"}

        filtered = []
        for f in funcs:
            params_str = str(f.parameters).lower()
            is_math = any(t in params_str for t in math_types)
            param_count = len(f.parameters)

            # Build metadata string for keyword matching
            metadata = (
                f"{f.name} {f.description} {' '.join(f.parameters.keys())}"
            ).lower()
            metadata_entities = set(re.findall(r'[a-z]{3,}', metadata))

            # Check for intersection between prompt and function description
            has_keywords = bool(
                (metadata_entities & prompt_entities) - stop_words
            )

            if is_math:
                # Math functions must match the exact number of digits found
                if num_count != param_count or num_count == 0:
                    continue
                if not has_keywords and len(prompt_entities) > 2:
                    continue
                filtered.append(f)
            elif has_keywords:
                filtered.append(f)
            elif (num_count == 0 and param_count == 1
                    and len(prompt_entities) <= 3):
                # Fallback for very short prompts
                if any(word in metadata for word in prompt_entities):
                    filtered.append(f)

        return filtered

    def generate(
        self, llm: Any, funcs: List["FunctionDefinition"], prompt: str
    ) -> str:
        """
        Executes constrained token generation to select a function.

        This method forces the LLM to choose from a list of valid function
        names by filtering the vocabulary at each step of the generation.

        Args:
            llm (Any): The LLM instance with encode and get_logits methods.
            funcs (List[FunctionDefinition]): List of possible functions.
            prompt (str): The user's natural language request.

        Returns:
            str: The name of the selected function or an empty string for none.
        """
        filtered_funcs = self.filter_functions(funcs, prompt)

        if not filtered_funcs:
            p_lower = prompt.lower()
            sql_kw = {"select", "insert", "update", "delete", "database"}
            interest_kw = {"interest", "rate", "principal", "compound"}
            filtered_funcs = [
                f for f in funcs
                if (f.name != "fn_execute_sql_query"
                    or any(kw in p_lower for kw in sql_kw))
                and (f.name != "fn_calculate_compound_interest"
                     or any(kw in p_lower for kw in interest_kw))
            ]

        # List of valid candidate strings
        f_names = sorted(list({f.name for f in filtered_funcs} | {"fn_none"}))
        instructions = self.get_func_instructions(filtered_funcs, prompt)
        output = "fn_"
        vocab = load_vocab(llm)

        # Constrained decoding loop
        while output not in f_names:
            candidates = [f for f in f_names if f.startswith(output)]
            if not candidates:
                return "fn_none"

            input_ids = llm.encode(instructions + output).tolist()[0]
            logits = llm.get_logits_from_input_ids(input_ids)

            # Filter vocabulary: only allow tokens that keep the string valid
            valid_token_map = {}
            for t_str, t_id in vocab.items():
                # Handle BPE space marker 'Ġ'
                clean_t = t_str[1:] if t_str.startswith("Ġ") else t_str
                if clean_t and any(
                    f.startswith(output + clean_t) for f in candidates
                ):
                    valid_token_map[t_id] = clean_t

            if not valid_token_map:
                return "fn_none"

            # Select the most likely token from the valid candidates
            best_id = max(valid_token_map.keys(), key=lambda x: logits[x])
            output += valid_token_map[best_id]

        return "" if output == "fn_none" else output
