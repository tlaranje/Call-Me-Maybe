*This project has been created as part of the 42 curriculum by \<tlaranje\>.*

# Call Me Maybe

## Description

This project implements a **function calling system** that translates natural language prompts into structured function calls. Given a prompt like `"What is the sum of 2 and 3?"`, the system does not answer `5` - instead it produces:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2.0,
    "b": 3.0
  }
}
```

The core challenge is reliability: small language models (0.6B parameters) are notoriously unreliable at generating structured output when prompted freely. This project solves that using **constrained decoding** - a technique that guides the model token-by-token, guaranteeing 100% valid and schema-compliant JSON output regardless of prompt complexity.


## Instructions

### Installation

Install dependencies

```bash
make install
```

### Running the project

```bash
make run
```

Or with custom paths:

```bash
make run \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Other Makefile targets

```bash
make lint           # Run flake8 and mypy
make lint-strict    # Run mypy --strict
make debug          # Run with Python debugger (pdb)
make clean          # Remove __pycache__ and .mypy_cache
make fclean         # Remove .llm, .uv_cache and .venv
```

## Algorithm Explanation

The system is built around two constrained decoding pipelines:

### 1. Function Name Selection

All available function names share the prefix `fn_`. Starting from that prefix, the algorithm:

1. Encodes the system prompt + current prefix into token IDs.
2. Gets the next-token logits from the LLM.
3. Filters the vocabulary to only tokens that extend the current prefix toward at least one valid function name.
4. Picks the highest-scoring valid token (greedy decoding).
5. Repeats until only one function name remains — at which point it completes without querying the model.

This guarantees the output is always a known function name.

### 2. Parameter Extraction

Parameters are generated progressively. For each parameter:

- The instruction prompt accumulates previously generated values as context (e.g. `a=2.0\nb=`), so the model sees what has already been extracted.
- **Numbers**: tokens are accepted only if appending them still produces a finite, valid `float`. Non-finite values (`nan`, `inf`) are rejected by setting their logits to `-inf`.
- **Strings**: any token is accepted except the newline token `Ċ`, which acts as the end-of-value signal. The BPE space marker `Ġ` is converted to a real space, preserving the original whitespace.


## Design Decisions

**Constrained number extraction**: numbers are generated token by token using the LLM. At each step, a token is accepted only if appending it to the current value still produces a finite, valid `float`. Tokens that break float parsing or produce `nan`/`inf` are rejected by setting their logit to `-math.inf`, forcing the model to try the next best token.

**Progressive context accumulation**: instead of creating a fresh prompt for each parameter, the assistant context grows with each generated value. This mirrors how the reference implementation works and significantly improves accuracy for multi-parameter functions.

**Single vocabulary load per generation**: `load_vocab` is called once per generator call, not once per token. This avoids repeatedly reading the vocabulary JSON file from disk.

**Pydantic for all input validation**: both prompt files and function definition files are validated with Pydantic models before any generation starts, providing clear error messages on malformed input.


## Performance Analysis

Tested on the provided `function_calling_tests.json` (11 prompts):

| Category | Accuracy |
|---|---|
| Function name selection | ~100% |
| Number extraction | ~100% |
| Simple string extraction (`name`, `s`) | ~90% |
| Multi-parameter string inference (`regex`) | ~80% |

**JSON validity**: 100% — every output is parseable and schema-compliant.

**Speed**: all 11 prompts process in under 1 minutes on standard CPU hardware.

The main accuracy limitation is the 0.6B model's ability to infer implicit regex patterns from natural language (e.g. inferring `\d+` from "replace all numbers"). Values explicitly present in the prompt are extracted correctly in nearly all cases.


## Challenges Faced

**Infinite generation loops**: the number generator would accumulate zeros indefinitely (e.g. `2.000000...`) because the model preferred digits after a decimal point. Solved by rejecting `nan`/`inf` values and neutralizing pure whitespace tokens (`Ġ`) with `-inf`.

**BPE space markers**: the tokenizer encodes spaces as `Ġ` prefixes on the following token. Stripping them naively caused whitespace to disappear from multi-word strings. Solved by converting `Ġ` to a real space when `curr_value` is non-empty.

**Multi-parameter positional extraction**: for prompts like `"What is the sum of 2 and 3?"`, the model tended to sum the numbers instead of extracting them positionally. Solved by accumulating the assistant context progressively so the model sees `a=2.0\nb=` before generating `b`.


## Testing Strategy

- Manual end-to-end testing with the provided `function_calling_tests.json` after each significant change.
- Visual inspection of the output JSON to verify schema compliance (correct keys, correct types).
- Edge cases tested: multi-parameter functions, strings with spaces, strings with quotes, large numbers, regex patterns.
- Compared output against the reference implementation from another project to identify whitespace handling bugs.


## Example Usage

```bash
# Default paths
make run

# Custom paths
make run \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

Example output (`data/output/function_calling_results.json`):

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": {
      "name": "shrek"
    }
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {
      "s": "'hello'"
    }
  }
]
```

---

## Resources

- [Qwen3 Model Card](https://huggingface.co/Qwen/Qwen3-0.6B) — the LLM used in this project
- [Constrained Decoding — Outlines library concepts](https://github.com/outlines-dev/outlines) — conceptual reference (library itself not used)
- [Pydantic documentation](https://docs.pydantic.dev/) — used for input validation
- [BPE Tokenization explained](https://huggingface.co/learn/nlp-course/chapter6/5) — background on how `Ġ` space markers work

### AI Usage

AI was used to understand some things about using the model in this project, such as how the model sees some character (e.g. "=", ":") and to refine docstrings and inline comments.