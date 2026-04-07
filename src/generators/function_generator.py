from typing import Any
from src.utils import load_vocab
import numpy as np

# if TYPE_CHECKING:
#     from llm_sdk import Small_LLM_Model as LLM_Model
#     from src.models import FunctionDefinition


class FunctionGenerator:
    def get_func_instructions(self, funcs: list, prompt: str) -> str:
        """Constrói o prompt de sistema para o seletor de funções."""
        func_descriptions = "\n".join(
            [f"- {f.name}: {f.description}" for f in funcs]
        )
        return (
            "<|im_start|>system\n"
            "You are a precise function selector.\n"
            "STRICT RULES:\n"
            "1. For math (add, multiply, product), use 'fn_add_numbers' "
            "or 'fn_multiply_numbers'.\n"
            "2. For files/paths, use 'fn_read_file'.\n"
            "3. NEVER use 'fn_execute_sql_query' for math or files.\n"
            "4. If no match exists, return 'fn_none'.\n\n"
            f"Available Functions:\n{func_descriptions}\n"
            "Selection: <|im_end|>\n"
            f"<|im_start|>user\n{prompt}\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def is_semantic_match(self, func_name: str, prompt: str) -> bool:
        """Valida se a função escolhida faz sentido para o prompt."""
        p = prompt.lower()
        if func_name == "fn_execute_sql_query":
            # Proteção contra falso positivo de ficheiros
            if any(x in p for x in ["/", "\\", ".", "file", "read"]):
                return False
            # Proteção contra falso positivo de matemática
            math_k = ["add", "sum", "multiply", "product", "calculate"]
            sql_k = ["select", "from", "where", "query", "database"]
            if any(m in p for m in math_k) and not any(s in p for s in sql_k):
                return False
        return True

    def get_probability(
        self, logits: Any, valid_token_ids: list[int]
    ) -> float:
        """Calcula a probabilidade softmax para os tokens válidos."""
        if not valid_token_ids:
            return 0.0
        valid_logits = np.array([logits[tid] for tid in valid_token_ids])
        exp_logits = np.exp(valid_logits - np.max(valid_logits))
        probs = exp_logits / exp_logits.sum()
        return float(np.max(probs))

    def generate(
        self, llm: Any, funcs: list, prompt: str, excluded: set | None = None
    ) -> str:
        """Gera o nome da função validando tokens e semântica."""
        if excluded is None:
            excluded = set()

        f_names = [f.name for f in funcs if f.name not in excluded]
        f_names_with_none = sorted(list(set(f_names) | {"fn_none"}))

        output = "fn_"
        vocab = load_vocab(llm)
        instructions = self.get_func_instructions(funcs, prompt)
        threshold = 0.7

        while output not in f_names_with_none:
            ft_list = [f for f in f_names_with_none if f.startswith(output)]
            if not ft_list:
                return ""
            if len(ft_list) == 1:
                output = ft_list[0]
                break

            ids = llm.encode(instructions + output).tolist()[0]
            logits = llm.get_logits_from_input_ids(ids)

            valid_tokens = []
            for s, tid in vocab.items():
                t_str = s[1:] if s.startswith("Ġ") else s
                if t_str and any(
                 f.startswith(output + t_str) for f in ft_list):
                    valid_tokens.append((tid, t_str))

            if not valid_tokens:
                return ""

            prob = self.get_probability(logits, [vt[0] for vt in valid_tokens])

            # Se a confiança for baixa, removemos a função atual da lista
            if prob < threshold and output != "fn_":
                return self.generate(llm, funcs, prompt, excluded | {output})

            _, best_tstr = max(valid_tokens, key=lambda x: logits[x[0]])
            output += best_tstr

        # Validação final: se for SQL mas o prompt for de ficheiro/math
        if output != "fn_none" and output != "":
            if not self.is_semantic_match(output, prompt):
                return self.generate(llm, funcs, prompt, excluded | {output})

        return "" if output == "fn_none" else output
