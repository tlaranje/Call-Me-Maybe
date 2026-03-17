#!/bin/bash

# PATHS
SGOINFRE_BASE="$HOME/sgoinfre"
if [ -d "$SGOINFRE_BASE" ]; then
    SGOINFRE="$SGOINFRE_BASE/Call-Me-Maybe"
else
    SGOINFRE="$(pwd)"
fi

export UV_CACHE_DIR="$SGOINFRE/.uv_cache"
export HF_HOME="$SGOINFRE/.llm"
export TRANSFORMERS_CACHE="$SGOINFRE/.llm"
