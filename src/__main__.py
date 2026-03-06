# from llm_sdk import Small_LLM_Model
from src.get_args import get_args
from src.validation_models import load_and_validate
from src.constants import Colors as C
from pydantic import ValidationError


def main() -> None:
    try:
        # model = Small_LLM_Model()
        print()
        args = get_args()
        data = load_and_validate(args)

        for item in data["prompts"]:
            prompt = item.prompt
            # input_ids = model.encode(prompt)
            # input_ids_list = input_ids[0].tolist()
            # logits = model.get_logits_from_input_ids(input_ids_list)
            print(f"Prompt: {prompt}")

        print()

        for item in data["functions"]:
            print(f"Function: {item.name}")
            print(f"Description: {item.description}")
            params = {k: v.type for k, v in item.parameters.items()}
            print(f"Parameters: {params}")
            print(f"Returns: {item.returns}\n")

        print()

        # print(f"\nNúmero de logits: {len(logits)}")
    except ValidationError as e:
        for error in e.errors():
            msg = error['msg'].removeprefix("Value error, ")
            print(f"{C.RED}{msg}{C.END}")
        exit()
    except Exception as e:
        print(f"{C.RED}{e}{C.END}")
        exit()


if __name__ == "__main__":
    main()
