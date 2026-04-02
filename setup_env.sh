#!/usr/bin/env bash

# Detect shell
if [ -n "$FISH_VERSION" ]; then
    SHELL_TYPE="fish"
else
    SHELL_TYPE="bash"
fi

# PATHS
SGOINFRE_BASE="$HOME/sgoinfre"
if [ -d "$SGOINFRE_BASE" ]; then
    SGOINFRE="$SGOINFRE_BASE/Call-Me-Maybe"
else
    SGOINFRE="$(pwd)"
fi

mkdir -p "$SGOINFRE"

VENV_PATH="$SGOINFRE/.venv"

# Create venv if needed
if [ ! -d "$VENV_PATH" ]; then
    python3 -m venv "$VENV_PATH"
fi

# 🔥 Export vars depending on shell
if [ "$SHELL_TYPE" = "fish" ]; then
    set -x UV_CACHE_DIR "$SGOINFRE/.uv_cache"
    set -x HF_HOME "$SGOINFRE/.llm"
    set -x TRANSFORMERS_CACHE "$SGOINFRE/.llm"
    set -x UV_ACTIVE 1
    set -x UV_LINK_MODE copy

    source "$VENV_PATH/bin/activate.fish"
else
    export UV_CACHE_DIR="$SGOINFRE/.uv_cache"
    export HF_HOME="$SGOINFRE/.llm"
    export TRANSFORMERS_CACHE="$SGOINFRE/.llm"

    export UV_ACTIVE=1
    export UV_LINK_MODE=copy

    . "$VENV_PATH/bin/activate"
fi