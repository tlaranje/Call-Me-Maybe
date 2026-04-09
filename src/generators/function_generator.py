from typing import Any, TYPE_CHECKING
from src.utils import load_vocab
from rich import print
import re

if TYPE_CHECKING:
    from src.models import FunctionDefinition


class FunctionGenerator:
    def get_func_instructions(
        self, funcs: list[FunctionDefinition], prompt: str
    ) -> str:
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
            f"Available Functions:\n{func_descriptions}\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n{prompt}\n<|im_end|>\n"
            "<|im_start|>assistant\nfn_"
        )

    def generate(
        self,
        llm: Any,
        funcs: list["FunctionDefinition"],
        prompt: str,
    ) -> str:
        p = prompt.lower()
        prompt_entities = set(re.findall(r'[a-z]{3,}|\d+', p))

        STOP_WORDS = {
            "the", "and", "with", "for", "this", "that",
            "from", "into", "all", "your", "using", "query"
        }

        filtered_funcs = []

        for f in funcs:
            func_metadata = f.name.lower() + " " + f.description.lower()
            for p_name in f.parameters.keys():
                func_metadata += " " + p_name.lower()

            metadata_entities = set(
                re.findall(r'[a-z]{3,}|\d+', func_metadata)
            )

            intersection = (metadata_entities & prompt_entities) - STOP_WORDS

            has_numbers = any(char.isdigit() for char in p)
            is_math_func = any(
                t in str(f).lower() for t in ["number", "integer", "float"]
            )

            if intersection:
                filtered_funcs.append(f)
                continue

            if is_math_func and not has_numbers:
                continue

            if is_math_func and has_numbers and len(prompt_entities) <= 6:
                filtered_funcs.append(f)
                continue

            if len(prompt_entities) <= 3 and len(f.parameters) <= 1:
                filtered_funcs.append(f)

        f_names = sorted(list({f.name for f in filtered_funcs} | {"fn_none"}))

        output = "fn_"
        vocab = load_vocab(llm)
        instructions = self.get_func_instructions(filtered_funcs, prompt)

        while output not in f_names:
            candidates = [f for f in f_names if f.startswith(output)]

            if not candidates:
                return "fn_none"

            ids = llm.encode(instructions + output).tolist()[0]
            logits = llm.get_logits_from_input_ids(ids)

            valid_token_map = {}
            for token_str, token_id in vocab.items():
                clean_token = (
                    token_str[1:] if token_str.startswith("Ġ") else token_str
                )
                if clean_token and any(
                     f.startswith(output + clean_token) for f in candidates):
                    valid_token_map[token_id] = clean_token

            if not valid_token_map:
                return "fn_none"

            best_token_id = max(
                valid_token_map.keys(), key=lambda tid: logits[tid]
            )
            output += valid_token_map[best_token_id]

        return "" if output == "fn_none" else output
