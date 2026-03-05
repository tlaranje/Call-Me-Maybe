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

# === VARIABLES ===
SETUP_ENV       := . ./setup_env.sh &&
DEFAULT_INPUT   := data/input/function_calling_tests.json
DEFAULT_OUTPUT  := data/output/function_calling_result.json

# === COMMANDS ===
RM              := rm -rf
FIND            := find

# === BUILD TARGETS ===
install:
	@clear && $(SETUP_ENV) uv sync

run:
	@clear && $(SETUP_ENV) uv run python -m src \
	    --input $(DEFAULT_INPUT) \
	    --output $(DEFAULT_OUTPUT) $(ARGS)

debug:
	@clear && $(SETUP_ENV) uv run python -m pdb $(MAIN)

clean:
	@clear
	@echo "Cleaning project cache..."
	@$(FIND) . -type d -name "__pycache__" -exec $(RM) {} +
	@$(FIND) . -type d -name ".mypy_cache" -exec $(RM) {} +
	@$(FIND) . -type d -name ".pytest_cache" -exec $(RM) {} +
	@$(FIND) . -type f -name "*.pyc" -delete
	@$(FIND) . -type f -name "*.pyo" -delete
	@echo "Removing sgoinfre cache..."
	@bash -c '. ./setup_env.sh && \
	rm -rf $$SGOINFRE/.venv $$SGOINFRE/.uv_cache $$SGOINFRE/.llm'

lint:
	@clear && $(SETUP_ENV) uv run flake8 .
	@$(SETUP_ENV) uv run mypy .

lint-strict:
	@clear && $(SETUP_ENV) flake8 .
	@$(SETUP_ENV) mypy . --strict

.PHONY: install run debug clean lint lint-strict