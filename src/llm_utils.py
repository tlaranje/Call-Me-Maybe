from llm_sdk import Small_LLM_Model as LLM_Model
from src.validation_models import FunctionDefinition as FuncDef
from src.utils import load_vocab


def generate_numbers(
    model: LLM_Model, func: FuncDef, prompt: str
) -> dict[str, int]:
    output = {}
    instructions = (
        f'<|im_start|>system\n'
        f'You are a params generate assistant.<|im_end|>\n'
        f'<|im_start|>user\n'
        f'In this function: {func}\n'
        f'Generate the params values '
        f'for this prompt is: "{prompt}"?<|im_end|>\n'
        f'<|im_start|>assistant\n'
        f'str:int'
    )
    vocab = load_vocab(model)
    for param_name, param in func.parameters.items():
        current_value = ""
        while True:
            input_ids = model._encode(instructions + current_value).tolist()[0]
            logits = model.get_logits_from_input_ids(input_ids)
            token_id = logits.index(max(logits))
            token_str = vocab[token_id]
            print(token_str)
            print(output)
            if token_str in [",", "}"]:
                break
            try:
                float(current_value + token_str)
                current_value += token_str
            except ValueError:
                pass
        output[param_name] = float(current_value)
    return output
