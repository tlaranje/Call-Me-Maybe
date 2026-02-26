from llm_sdk import Small_LLM_Model


def main():
    model = Small_LLM_Model()

    text = "The capital of France is"
    input_ids = model.encode(text)

    # Converter tensor -> lista de ints
    input_ids_list = input_ids[0].tolist()

    logits = model.get_logits_from_input_ids(input_ids_list)

    print("Número de logits:", len(logits))
    print("Primeiros 10 logits:", logits[:10])


if __name__ == "__main__":
    main()
