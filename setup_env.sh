#!/bin/bash

# PATHS
SGOINFRE_BASE="$HOME/sgoinfre"
if [ -d "$SGOINFRE_BASE" ]; then
    SGOINFRE="$SGOINFRE_BASE/Call-Me-Maybe"
else
    SGOINFRE="$(pwd)"
fi

mkdir -p "$SGOINFRE"

export UV_CACHE_DIR="$SGOINFRE/.uv_cache"
export HF_HOME="$SGOINFRE/.llm"
export TRANSFORMERS_CACHE="$SGOINFRE/.llm"

export UV_ACTIVE=1
export UV_LINK_MODE=copy

VENV_PATH="$SGOINFRE/.venv"

if [ ! -d "$VENV_PATH" ]; then
    python3 -m venv "$VENV_PATH"
fi

. "$VENV_PATH/bin/activate"
