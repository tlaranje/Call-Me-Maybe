# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Makefile                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: tlaranje <tlaranje@student.42porto.com>    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2026/02/26 12:06:51 by tlaranje          #+#    #+#              #
#    Updated: 2026/04/06 10:40:48 by tlaranje         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

# === VARIABLES ===
DEFAULT_INPUT		:= data/input/function_calling_tests.json
DEFAULT_DEFINITION	:= data/input/functions_definition.json
DEFAULT_OUTPUT		:= data/output/function_calling_result.json

RM					:= rm -rf
FIND				:= find

# === DIRENV SETUP ===
DIRENV_BIN := $(HOME)/.local/bin/direnv

define SETUP_DIRENV
if ! command -v direnv >/dev/null 2>&1; then \
	echo "Installing direnv..."; \
	curl -sfL https://direnv.net/install.sh | bash; \
	export PATH="$(HOME)/.local/bin:$$PATH"; \
fi; \
direnv allow >/dev/null 2>&1 || true
endef

CLEAR := $(SETUP_DIRENV) && clear

# === BUILD TARGETS ===
install:
	@$(CLEAR) && uv sync

INPUT ?= $(DEFAULT_INPUT)
OUTPUT ?= $(DEFAULT_OUTPUT)
FUNCS ?= $(DEFAULT_DEFINITION)

run:
	@$(CLEAR) && uv run python -m src \
	--input $(INPUT) \
	--functions_definition $(FUNCS) \
	--output $(OUTPUT)

debug:
	@$(CLEAR) && uv run python -m pdb -m src \
	--input $(DEFAULT_INPUT) \
	--functions_definition $(DEFAULT_DEFINITION) \
	--output $(DEFAULT_OUTPUT) $(ARGS)

clean:
	@clear
	@echo "Cleaning project cache..."
	@$(FIND) . -type d -name "__pycache__" -exec $(RM) {} +
	@$(FIND) . -type d -name ".mypy_cache" -exec $(RM) {} +
	@$(FIND) . -type d -name ".pytest_cache" -exec $(RM) {} +
	@$(FIND) . -type f -name "*.pyc" -delete
	@$(FIND) . -type f -name "*.pyo" -delete

fclean: clean
	@echo "Removing sgoinfre cache..."
	@rm -rf $(HOME)/sgoinfre/Call-Me-Maybe/.venv \
	        $(HOME)/sgoinfre/Call-Me-Maybe/.uv_cache \
	        $(HOME)/sgoinfre/Call-Me-Maybe/.llm

lint:
	@clear && uv run flake8 .
	@uv run mypy . --warn-return-any \
	    --warn-unused-ignores \
	    --ignore-missing-imports \
	    --disallow-untyped-defs \
	    --check-untyped-defs

lint-strict:
	@clear && uv run flake8 .
	@uv run mypy . --strict

.PHONY: install run debug clean lint lint-strict