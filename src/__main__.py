from llm_sdk import Small_LLM_Model, AutoTokenizer
from src.get_args import get_args
from src.json_utils import load_json_file
import json
import sys
import os


def main():
    try:
        model = Small_LLM_Model()
        print()

        args_dict = get_args()
        data = load_json_file(args_dict['Input'])
        prompts = [item['prompt'] for item in data]
        text_data = " ".join(prompts)

        input_ids = model.encode(text_data)
        input_ids_list = input_ids[0].tolist()
        logits = model.get_logits_from_input_ids(input_ids_list)

        print("Número de logits:", len(logits))
    except Exception as e:
        print(f"{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
