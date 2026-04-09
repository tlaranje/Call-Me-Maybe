*This project has been created as part of the 42 curriculum by \<tlaranje\>.*

# Call Me Maybe

## Description

**Call Me Maybe** is a function calling system that translates natural language prompts into structured, schema-compliant function calls using a small local LLM (Qwen3-0.6B). Given a prompt like `"What is the sum of 2 and 3?"`, it does not return `5` — instead it produces:

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

The core challenge is reliability: a 0.6B parameter model is notoriously unreliable at producing structured output when prompted freely. This project solves that using **constrained decoding** — a technique that intercepts the model's token generation at each step, masking any token that would break the target schema, guaranteeing 100% valid and parseable JSON output regardless of prompt complexity.



## Instructions

### Requirements

- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) package manager
- The `llm_sdk/` package (copy it to the project root alongside `src/`)

### Installation

```bash
make install
```

This runs `uv sync`, which creates a virtual environment and installs all dependencies from `pyproject.toml`.

### Running the project

```bash
make run
```

Or with explicit paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

All three arguments are optional — the paths above are the defaults.

### Other Makefile targets

```bash
make lint           # flake8 + mypy with standard flags
make lint-strict    # flake8 + mypy --strict
make debug          # Run with Python's built-in debugger (pdb)
make clean          # Remove __pycache__ and .mypy_cache
make fclean         # Remove .llm, .uv_cache and .venv
```


## Algorithm Explanation

The pipeline is split into two constrained decoding stages.

### Stage 1 — Function Name Selection (`FunctionGenerator`)

Function names all share the prefix `fn_`. The generator builds the name character by character:

1. Start with the fixed prefix `fn_`.
2. Encode `system_prompt + current_prefix` into token IDs and call the LLM to get logits.
3. Filter the vocabulary to tokens whose string value extends the current prefix toward at least one valid function name.
4. Pick the highest-scoring valid token (greedy decoding) and append it.
5. Repeat until only one candidate remains — then complete it without another LLM call.

A confidence check runs at each step: if the softmax probability of the valid tokens falls below a threshold (0.9), the current branch is discarded and generation restarts with that function name excluded. A secondary semantic validator (`it_match`) applies rule-based guardrails to prevent common confusions (e.g. regex vs. template formatting, SQL vs. basic arithmetic).

### Stage 2 — Parameter Extraction (`FunctionCaller` + type generators)

Parameters are generated one at a time. The assistant context grows progressively, so the model always sees previously extracted values before generating the next one (e.g. `a=2.0\nb=`).

Each parameter type has a dedicated constrained generator:

- **Numbers** (`Numbers`): Tokens are accepted only if appending them keeps the accumulated string a valid numeric prefix (handles integers, floats, and scientific notation). Tokens producing `nan`, `inf`, or an invalid float are masked to `-inf`.
- **Strings** (`String`): Any token is accepted except the newline token `Ċ`, which signals end-of-value. The BPE space marker `Ġ` is converted to a real space to preserve whitespace in multi-word strings.
- **Booleans** (`Boolean`): Tokens are accepted only if they form a prefix of `"true"` or `"false"`. If a chosen token leads to a dead end, its logit is set to `-inf` and the generator backtracks.


## Design Decisions

**Progressive context accumulation** — rather than building a new prompt per parameter, the assistant context is extended with each extracted value. This significantly improves accuracy for multi-parameter functions because the model can use already-generated values as anchors.

**Single vocabulary load per generation call** — `load_vocab` is called once per generator invocation, not once per token, avoiding repeated disk reads of the vocabulary JSON.

**Pydantic for all input validation** — `FunctionDefinition`, `FunctionParameter`, and `Prompt` are all Pydantic models with custom `model_validator` hooks. Any malformed input file produces a clear, structured error message before generation starts.

**Confidence-based function backtracking** — if the model is unsure which function to pick (low softmax probability on valid tokens), the current candidate is excluded and generation restarts. This avoids silently returning the wrong function.

**Semantic guardrails in `it_match`** — certain function confusions are common enough to warrant explicit rules: regex substitution vs. template formatting, SQL queries vs. basic math. These are checked after the constrained decoding step.


## Performance Analysis

Tested on the provided `function_calling_tests.json` (11 prompts, CPU hardware):

| Category | Accuracy |
|---|---|
| Function name selection | ~100% |
| Number extraction | ~100% |
| Simple string extraction | ~95% |
| Multi-parameter / regex inference | ~90% |

**JSON validity**: 100% — every output is parseable and schema-compliant by construction.

**Speed**: all 11 prompts processed in under 2~ minute on standard CPU hardware.

The main accuracy limitation is the 0.6B model's ability to infer implicit values from ambiguous prompts (e.g. deriving a regex pattern from a natural language description). Values explicitly stated in the prompt are extracted correctly in nearly all cases.



## Challenges Faced

**Infinite number loops** — the number generator accumulated digits indefinitely (e.g. `2.000000...`) because the model preferred extending decimal places. Fixed by rejecting tokens that produce `nan` or `inf` and by neutralising pure whitespace tokens.

**BPE space markers** — the tokenizer prefixes space-bearing tokens with `Ġ`. Stripping it naively removed whitespace from multi-word strings. Fixed by converting `Ġ` to a real space only when the current accumulated value is non-empty.

**Multi-parameter positional confusion** — for prompts like `"What is the sum of 2 and 3?"`, the model tended to compute the result (`5`) instead of extracting both operands separately. Fixed by the progressive context accumulation strategy, which shows `a=2.0\nb=` before the model generates `b`.

**Wrong function selection for edge cases** — early versions confused regex substitution with template formatting on prompts mentioning placeholders. Addressed via the `it_match` semantic validator.



## Testing Strategy

- End-to-end runs with the provided `function_calling_tests.json` after each significant change.
- Visual inspection of the output JSON for schema compliance (correct keys, correct types, no extra fields).
- Edge cases tested: multi-parameter functions, strings with spaces and quotes, large numbers, scientific notation, regex patterns, boolean values.
- Output compared against expected results to catch regressions during refactoring.



## Example Usage

```bash
# Run with default paths
make run

# Run with explicit paths
uv run python -m src \
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
    "parameters": { "a": 2.0, "b": 3.0 }
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": { "name": "shrek" }
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": { "s": "hello" }
  }
]
```

## Resources

- [Qwen3 Model Card](https://huggingface.co/Qwen/Qwen3-0.6B) - the LLM used in this project
- [Softmax Activation Function in Neural Networks](https://www.geeksforgeeks.org/deep-learning/the-role-of-softmax-in-neural-networks-detailed-explanation-and-applications/) - used to understand softmax probability calculation for valid token selection
- [Pydantic documentation](https://docs.pydantic.dev/) - used for all input validation
- [HuggingFace NLP Course - BPE Tokenization](https://huggingface.co/learn/nlp-course/chapter6/5) - background on how `Ġ` space markers work in BPE vocabularies

### AI Usage

AI was used to clarify how the Qwen tokenizer encodes specific characters (e.g. `=`, `:`, `Ġ`, `Ċ`), to understand the structure of the vocabulary JSON file, and to refine docstrings and inline comments throughout the codebase.