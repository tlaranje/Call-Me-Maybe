# from llm_sdk import Small_LLM_Model
from src.get_args import get_args
from src.json_utils import load_json_file
import json
import sys


def main():
    try:
        args_dict = get_args()
        data = load_json_file(args_dict['Input'])

        print("Output:")
        print(json.dumps(data, indent=4, ensure_ascii=False))
    except Exception as e:
        print(f"{e}")
        sys.exit(1)
    """ model = Small_LLM_Model()
    print("\n\n")

    text = "The capital of France is"
    input_ids = model.encode(text)

    # Converter tensor -> lista de ints
    input_ids_list = input_ids[0].tolist()

    logits = model.get_logits_from_input_ids(input_ids_list)

    print("Número de logits:", len(logits))
    print("Primeiros 10 logits:", logits[:10]) """


if __name__ == "__main__":
    main()
