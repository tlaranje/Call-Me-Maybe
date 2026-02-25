from llm_sdk.small_llm_model import Small_LLM_Model

if __name__ == "__main__":
    model = Small_LLM_Model()
    input_ids = model.encode("Hello world!")
    logits = model.get_logits_from_input_ids(input_ids)

    # logits é um tensor com shape [1, vocab_size]
    logits = logits[0]  # remove batch dimension

    # aplicar restrições
    for token_id in range(vocab_size):
        if token_id not in allowed_tokens:
            logits[token_id] = -float("inf")

    # escolher o token com maior logit
    next_token = argmax(logits)
