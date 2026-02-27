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

# === PATHS ===
SGOINFRE_BASE := /home/$(USER)/sgoinfre

ifeq ($(wildcard $(SGOINFRE_BASE)),)
	SGOINFRE := $(CURDIR)
else
	SGOINFRE := $(SGOINFRE_BASE)/Call-Me-Maybe
endif

HF_CACHE := $(SGOINFRE)/.llm
UV_CACHE := $(SGOINFRE)/.uv_cache
VENV     := $(SGOINFRE)/.venv

# VARIABLES
DEFAULT_INPUT	:=	data/input/function_calling_tests.json
DEFAULT_OUTPUT	:=	data/output/function_calling_result.json

# === COMMANDS ===
RM				:= rm -rf
FIND			:= find

# === EXPORTS ===
export UV_PROJECT_ENVIRONMENT=$(VENV)
export UV_CACHE_DIR=$(UV_CACHE)
export HF_HOME=$(HF_CACHE)
export TRANSFORMERS_CACHE=$(HF_CACHE)
export HF_TOKEN="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"

# === BUILD TARGETS ===
install:
	@pip install uv --break-system-packages
	@clear && uv sync

run:
	@clear && export HF_TOKEN="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" && \
	uv run python -m src --input $(DEFAULT_INPUT) --output $(DEFAULT_OUTPUT) \
	$(ARGS)

debug:
	@clear && uv run python -m pdb $(MAIN)

clean:
	@clear
	@echo "Cleaning project cache..."
	@$(FIND) . -type d -name "__pycache__" -exec $(RM) {} +
	@$(FIND) . -type d -name ".mypy_cache" -exec $(RM) {} +
	@$(FIND) . -type d -name ".pytest_cache" -exec $(RM) {} +
	@$(FIND) . -type f -name "*.pyc" -delete
	@$(FIND) . -type f -name "*.pyo" -delete
	@echo "Removing sgoinfre environment..."
	@$(RM) $(VENV)
	@$(RM) $(UV_CACHE)
	@$(RM) $(HF_CACHE)

lint:
	@clear
	@uv run flake8 .
	@uv run mypy .

lint-strict:
	@clear
	@flake8 .
	@mypy . --strict

.PHONY: install run debug clean fclean lint