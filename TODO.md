# call me maybe — Guia Passo a Passo (sem código pronto)

---

## Visão geral do que tens de construir

O programa recebe dois ficheiros JSON:
- uma lista de prompts em linguagem natural
- uma lista de definições de funções (nome, parâmetros, tipos, descrição)

Para cada prompt, o programa tem de devolver qual função chamar e com que argumentos,
em formato JSON. O truque é que não podes deixar o modelo gerar livremente — tens de
guiar a geração token a token para garantir JSON 100% válido.

---

## Os ficheiros que tens de implementar, por ordem

---

### 1. `src/get_args.py` — argumentos da linha de comandos

**O que faz:** Lê os argumentos passados ao programa quando é executado.

**Como pensar:**
- Usa `argparse` para definir 3 argumentos: `--input`, `--output`, `--functions_definition`
- Para cada um define um valor padrão que aponte para os ficheiros em `data/input/` e `data/output/`
- Atenção: o PDF usa `--functions_definition` no singular — confirma que o teu código usa exatamente esse nome
- A função deve devolver um dicionário simples com as 3 chaves

**Perguntas para te guiar:**
- Que acontece se o utilizador não passar nenhum argumento? (deve usar os defaults)
- Que nome de chave vais usar no dicionário de retorno para não confundir com o nome do argumento?

---

### 2. `src/json_utils.py` — carregar ficheiros JSON

**O que faz:** Lê um ficheiro JSON do disco e devolve o conteúdo.

**Como pensar:**
- Abre o ficheiro com um context manager (`with open(...)`)
- Usa `json.load()` para fazer o parse
- Trata dois erros possíveis:
  - O ficheiro não existe → `FileNotFoundError`
  - O conteúdo não é JSON válido → `json.JSONDecodeError`
- Em ambos os casos, lança uma exceção com uma mensagem clara para o utilizador

**Perguntas para te guiar:**
- Qual o tipo de retorno? (pode ser qualquer coisa — lista, dict, etc.)
- Como é que o `__main__.py` vai saber se algo correu mal?

---

### 3. `src/validation_models.py` — modelos Pydantic

**O que faz:** Define as classes de validação e carrega/valida os ficheiros de input.

**Classes a definir:**

**`FunctionParameter`**
- Um parâmetro tem apenas um campo: `type` (string)

**`FunctionDefinition`**
- Tem: `name`, `description`, `parameters` (dicionário de nome → FunctionParameter), `returns`
- Adiciona um `@model_validator` que verifica se todos os campos obrigatórios existem e têm o tipo certo
- Se houver erros, junta-os todos numa única mensagem (não para ao primeiro erro)

**`Prompt`**
- Tem apenas um campo: `prompt` (string)
- Valida que existe e é uma string

**Função `load_and_validate(args)`**
- Carrega os dois ficheiros JSON usando `json_utils.py`
- Para cada item em cada ficheiro, tenta criar o modelo Pydantic correspondente
- Recolhe todos os erros (não para ao primeiro)
- Se houver qualquer erro, lança uma exceção com todos eles listados
- Se tudo estiver bem, devolve um dicionário com a lista de prompts e a lista de funções

**Perguntas para te guiar:**
- O que acontece se o ficheiro tiver 10 prompts e 2 forem inválidos? Deves parar ou continuar e reportar todos?
- Como distingues erros do ficheiro de prompts dos erros do ficheiro de funções na mensagem de erro?

---

### 4. `src/llm_utils.py` — o núcleo do projeto

Este é o ficheiro mais importante. Tem 4 funções distintas.

---

#### 4a. `load_vocab(model)`

**O que faz:** Carrega o vocabulário do modelo — a tabela que mapeia cada ID de token
para a sua representação em texto.

**Como pensar:**
- Chama `model.get_path_to_vocab_file()` para obter o caminho do ficheiro
- Abre e lê o ficheiro JSON — é um dicionário `{token_string: token_id}`
- Tens de **inverter** esse dicionário para ficar com `{token_id: token_string}`
- Devolve esse dicionário invertido

**Porquê inverter?** Porque durante a geração recebes IDs e precisas de saber a que
string correspondem. O ficheiro original está na direção contrária.

---

#### 4b. `build_prompt(functions, prompt)`

**O que faz:** Constrói o texto completo que vai ser dado ao modelo como input.

**Como pensar:**
- O modelo precisa de contexto para escolher a função certa
- O texto deve incluir: a lista de funções disponíveis (nome, descrição, parâmetros) e o prompt do utilizador
- Deve também indicar ao modelo o formato de resposta esperado
- Experimenta diferentes formulações — a qualidade do prompt afeta a escolha da função

**Perguntas para te guiar:**
- Em que formato mostras as funções disponíveis? (JSON? texto? lista?)
- Como indicas ao modelo que deve responder apenas com JSON e nada mais?
- O prompt deve vir antes ou depois das definições de funções?

---

#### 4c. `get_valid_token_ids(vocab, partial_json, functions)`

**Esta é a função mais difícil e mais importante do projeto.**

**O que faz:** Dado o JSON gerado até agora, devolve a lista de IDs de tokens que são
válidos para o próximo passo.

**Como pensar — abordagem por fases:**

Antes de implementar, desenha no papel todas as fases possíveis da geração do JSON.
O JSON final que queres gerar tem sempre esta estrutura:

```
{"name": "<nome_da_função>", "parameters": {<pares chave-valor>}}
```

Cada fase corresponde a um ponto diferente nessa string:

**Fase 1 — Início**
O JSON está vazio. Qual é o único token válido?

**Fase 2 — Após `{`**
Estamos a gerar `"name"`. Quais tokens podem vir aqui?

**Fase 3 — Após `"name"`**
Precisamos de `: "`. Quais tokens formam isso?

**Fase 4 — A gerar o nome da função**
Aqui está o truque principal. Tens a lista de nomes de funções válidos. O token seguinte
só é válido se o nome parcial que estás a gerar ainda pode vir a ser um dos nomes válidos.

Exemplo: se tens `fn_a` gerado até agora, o token `dd` é válido (porque `fn_add` existe),
mas o token `xyz` não é válido.

Como verificas isso? Para cada token no vocabulário, concatenas ao que já foi gerado
e verificas se algum nome de função **começa com** esse resultado.

**Fase 5 — Após o nome estar completo**
Precisas de `, "parameters": {`. Quais tokens são válidos aqui?

**Fase 6 — A gerar os parâmetros**
Para cada parâmetro, tens de gerar a chave (nome do parâmetro) e o valor com o tipo certo.
- Para parâmetros do tipo `number`: só tokens que formam dígitos, ponto decimal, sinal negativo
- Para parâmetros do tipo `string`: qualquer conteúdo entre aspas
- Para parâmetros do tipo `boolean`: apenas `true` ou `false`
- Tens de saber quando um parâmetro termina e começa o próximo (vírgulas)
- Tens de saber quando todos os parâmetros estão preenchidos (fecho com `}`)

**Fase 7 — Fim**
Todos os parâmetros estão preenchidos. O único token válido é `}`.

**Como determinar em que fase estás:**
Analisa a string `partial_json` que já tens. Podes usar `re` (regex) para extrair
o que já foi gerado. Pergunta-te: "O que é que esta string já contém? O que falta?"

**Dica sobre os tokens do vocabulário BPE:**
Um token pode ser uma letra, uma palavra inteira, ou até um fragmento como `fn_add`.
Quando verificas se um token é válido, tens de pensar em prefixos — não apenas em
caracteres individuais.

**Perguntas para te guiar:**
- Como sabes em que fase estás apenas olhando para `partial_json`?
- Como tratas o caso em que o nome da função ainda está a ser construído?
- Como sabes quais parâmetros já foram gerados e quais faltam?
- O que acontece quando há múltiplos parâmetros? Em que ordem os generates?

---

#### 4d. `generate_function_call(model, vocab, functions, prompt)`

**O que faz:** Junta tudo — executa o loop de geração token a token.

**Como pensar — o algoritmo principal:**

```
1. Constrói o prompt completo com build_prompt()
2. Codifica o prompt em input_ids com model.encode()
3. Inicializa uma lista vazia para os tokens gerados e uma string vazia para o JSON parcial
4. Repete até o JSON estar completo (ou atingir um máximo de tokens):
   a. Chama model.get_logits_from_input_ids() com os input_ids atuais
   b. Chama get_valid_token_ids() para saber quais tokens são válidos agora
   c. Para todos os tokens NÃO válidos, põe o seu logit a -inf (menos infinito)
   d. Escolhe o token com o logit mais alto (argmax)
   e. Adiciona esse token à lista de gerados e ao JSON parcial
   f. Verifica se o JSON parcial já está completo e válido
5. Faz json.loads() do JSON gerado
6. Devolve o resultado
```

**Como verificar se o JSON está completo:**
Tenta fazer `json.loads()` do que foi gerado. Se não lançar exceção, está completo.
Podes também verificar se começa com `{` e termina com `}` antes de tentar o parse.

**Dica sobre input_ids:**
O modelo precisa de ver TODO o contexto gerado até agora. A cada iteração, os input_ids
que passas ao modelo devem ser os IDs do prompt original + todos os tokens que já geraste.

**Perguntas para te guiar:**
- O que pões como limite máximo de tokens? (considera o caso de a geração ficar presa)
- Como tratas o caso em que `get_valid_token_ids` devolve uma lista vazia?
- Que informação guardas para construir o resultado final (`name` e `parameters`)?

---

### 5. `src/__main__.py` — juntar tudo

**O que faz:** Entry point do programa — chama tudo pela ordem certa.

**Como pensar — ordem de operações:**

```
1. Carregar o modelo (Small_LLM_Model)
2. Carregar o vocabulário (load_vocab)
3. Ler os argumentos (get_args)
4. Carregar e validar os ficheiros de input (load_and_validate)
5. Para cada prompt na lista:
   a. Chamar generate_function_call()
   b. Construir o objeto de resultado {prompt, name, parameters}
   c. Adicionar à lista de resultados
6. Escrever a lista de resultados no ficheiro de output
7. Tratar exceções em cada passo com mensagens claras
```

**Cuidados importantes:**
- Cria o diretório de output se não existir antes de tentar escrever
- Cada prompt é processado individualmente — não juntes todos num único call ao modelo
- O programa nunca deve crashar sem mostrar uma mensagem útil ao utilizador
- O ficheiro de output deve ser um array JSON (lista de objetos)

---

## Ordem recomendada para implementar e testar

1. `get_args.py` e `json_utils.py` — são simples e testáveis imediatamente com prints
2. `validation_models.py` — testa carregando os ficheiros de exemplo e verificando os erros
3. `load_vocab` e `build_prompt` — faz print do vocabulário e do prompt para verificar
4. O loop de geração **sem** constrained decoding primeiro — deixa o modelo gerar livremente e vê o que produz (vai ser mau, mas confirma que o loop funciona)
5. `get_valid_token_ids` progressivamente:
   - Começa por só enforçar a estrutura base do JSON (só `{` no início)
   - Depois enforça o campo `"name"` com valores válidos
   - Depois enforça os tipos dos parâmetros
6. Junta tudo no `__main__.py` e testa com os ficheiros de exemplo

---

## Bugs conhecidos no código atual para corrigir

- `--functions_definition` vs `--functions_definitions`: o PDF usa o singular
- O output por default aponta para `function_calling_result.json` — devia ser `function_calling_results.json` (com 's' no fim)
- O `__main__.py` atual não itera sobre os prompts individualmente
- O `__main__.py` não escreve nada no ficheiro de output

---

## Recursos para aprender os conceitos necessários

### Constrained Decoding — o conceito central

- **Artigo "Guidance" (Microsoft)** — teoria de guiar geração de LLMs:
  https://arxiv.org/abs/2307.09702

- **Blog "Structured Outputs" (Hugging Face)**:
  https://huggingface.co/blog/structured-generation

- **Repositório lm-format-enforcer** — implementação de referência, só para estudar o README:
  https://github.com/noamgat/lm-format-enforcer

### Como os LLMs geram texto (logits, tokens, argmax)

- **Andrej Karpathy — "Let's build GPT from scratch"** (YouTube — essencial):
  https://www.youtube.com/watch?v=kCc8FmEb1nY

- **HuggingFace — Decoding Strategies**:
  https://huggingface.co/blog/how-to-generate

### Tokenização e vocabulário BPE

- **Andrej Karpathy — "Let's build the GPT tokenizer"** (YouTube):
  https://www.youtube.com/watch?v=zduSFxRajkE

- **HuggingFace — Summary of Tokenizers**:
  https://huggingface.co/docs/transformers/tokenizer_summary

### Function Calling — o conceito do projeto

- **OpenAI — Function Calling guide** (boa introdução ao conceito):
  https://platform.openai.com/docs/guides/function-calling

- **Artigo "Gorilla"** — benchmark de function calling em LLMs:
  https://arxiv.org/abs/2305.15334

### Pydantic v2 — validação de dados

- **Documentação oficial — Models**:
  https://docs.pydantic.dev/latest/concepts/models/

- **Documentação oficial — Validators**:
  https://docs.pydantic.dev/latest/concepts/validators/

### Ferramentas Python usadas no projeto

- **Módulo `argparse`**: https://docs.python.org/3/library/argparse.html
- **Módulo `json`**: https://docs.python.org/3/library/json.html
- **Módulo `re` (regex)**: https://docs.python.org/3/library/re.html

---

## Checklist antes de submeter

- [ ] `make install` corre sem erros
- [ ] `make run` produz `data/output/function_calling_results.json`
- [ ] O output é um array JSON válido (valida em https://jsonlint.com)
- [ ] Cada objeto no output tem exatamente as chaves: `prompt`, `name`, `parameters`
- [ ] Os tipos dos valores em `parameters` correspondem à definição (`number` → float, `string` → str)
- [ ] `make lint` passa sem erros (flake8 + mypy)
- [ ] O argumento `--functions_definition` (singular) é aceite
- [ ] O programa nunca crasha — trata sempre os erros com mensagens claras
- [ ] O diretório `data/output/` não está no repositório (está no `.gitignore`)
- [ ] O `README.md` tem todas as secções obrigatórias do PDF
- [ ] Nenhum ficheiro em `src/` importa `torch` ou `transformers` diretamente