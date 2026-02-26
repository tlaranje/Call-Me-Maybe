# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Makefile                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: tlaranje <tlaranje@student.42porto.com>    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2026/02/26 12:06:51 by tlaranje          #+#    #+#              #
#    Updated: 2026/02/26 12:06:52 by tlaranje         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

# === Paths ===
SGOINFRE    := /home/$(USER)/sgoinfre/Call_Me_Maybe
VENV        := $(SGOINFRE)/.venv
UV_CACHE    := $(SGOINFRE)/.uv_cache
HF_CACHE    := $(SGOINFRE)/.llm

# === Main script ===
MAIN        := srcs/main.py

# === Commands ===
P3          := python3
RM          := rm -rf
FIND        := find

# === Build targets ===
install:
	clear
	@echo ">> Creating virtual environment... <<"
	@echo ""
	@mkdir -p $(VENV) $(UV_CACHE) $(HF_CACHE)
	@uv venv $(VENV)
	@UV_PROJECT_ENVIRONMENT=$(VENV) UV_CACHE_DIR=$(UV_CACHE) HF_HOME=$(HF_CACHE) TRANSFORMERS_CACHE=$(HF_CACHE) uv sync
	@UV_PROJECT_ENVIRONMENT=$(VENV) UV_CACHE_DIR=$(UV_CACHE) HF_HOME=$(HF_CACHE) TRANSFORMERS_CACHE=$(HF_CACHE) uv pip install flake8 mypy

run:
	clear
	@export UV_PROJECT_ENVIRONMENT=$(VENV) && \
	export UV_CACHE_DIR=$(UV_CACHE) && \
	export HF_HOME=$(HF_CACHE) && \
	export TRANSFORMERS_CACHE=$(HF_CACHE) && \
	uv run python $(MAIN)

debug:
	@export UV_PROJECT_ENVIRONMENT=$(VENV) && \
	export UV_CACHE_DIR=$(UV_CACHE) && \
	export HF_HOME=$(HF_CACHE) && \
	export TRANSFORMERS_CACHE=$(HF_CACHE) && \
	uv run python -m pdb $(MAIN)

clean:
	clear
	@echo ">> Cleaning project cache... <<"
	@$(FIND) . -type d -name "__pycache__" -exec $(RM) {} +
	@$(FIND) . -type d -name ".mypy_cache" -exec $(RM) {} +
	@$(FIND) . -type d -name ".pytest_cache" -exec $(RM) {} +
	@$(FIND) . -type f -name "*.pyc" -delete
	@$(FIND) . -type f -name "*.pyo" -delete

fclean: clean
	@echo ">> Removing sgoinfre environment... <<"
	@$(RM) $(VENV)
	@$(RM) $(UV_CACHE)
	@$(RM) $(HF_CACHE)

lint:
	@export UV_PROJECT_ENVIRONMENT=$(VENV) && uv run flake8 .
	@export UV_PROJECT_ENVIRONMENT=$(VENV) && uv run mypy .

lint-strict:
	@flake8 .
	@mypy . --strict

.PHONY: install run debug clean fclean lint