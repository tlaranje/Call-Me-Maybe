from typing import TYPE_CHECKING
from src.utils import load_vocab
import math

if TYPE_CHECKING:
    from llm_sdk import Small_LLM_Model as LLM_Model


class String:
    def generate(self, llm: LLM_Model, ins: str) -> str:
        curr_value = ""
        curr_token = ""

        # Carregamos o vocabulário uma vez para evitar overhead
        vocab = load_vocab(llm)

        # STOP_TOKENS reduzido apenas ao terminador lógico (Newline/Ċ)
        # Removidos: ',' e '"' para permitir SQL e strings complexas
        STOP_TOKENS = {'Ċ'}

        # Limite de iterações para evitar loops infinitos caso o modelo alucine
        max_tokens = 100
        iterations = 0

        while iterations < max_tokens:
            input_ids = llm.encode(ins + curr_value).tolist()[0]
            logits = llm.get_logits_from_input_ids(input_ids)

            # Seleção de tokens (Greedy)
            raw_token = max(vocab.keys(), key=lambda s: logits[vocab[s]])

            # 1. Tratamento de Espaços BPE (Ġ)
            # Se começar com Ġ, convertemos para espaço real
            processed_token = raw_token.replace('Ġ', ' ')

            # 2. Verificação de parada (Newline)
            if any(stop in raw_token for stop in STOP_TOKENS):
                # Adiciona o que veio antes da quebra de linha, se houver
                curr_value += processed_token.split('\n')[0].split('Ċ')[0]
                break

            # 3. Limpeza de caracteres de escape e aspas residuais de parsing
            # Importante: não remova aspas do MEIO da string, apenas do início/fim se necessário
            # mas para parâmetros de função, geralmente queremos manter a integridade.

            curr_value += processed_token
            iterations += 1

        # Limpeza final: removemos aspas externas que o LLM possa ter gerado por engano
        # mas mantemos as internas (como em 'INSERT INTO...')
        return curr_value.strip()
