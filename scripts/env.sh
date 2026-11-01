#!/usr/bin/env bash
# Source this file before installing extensions or running an experiment.
export HYBRIDREPRE_WORKDIR="${1:-${HYBRIDREPRE_WORKDIR:-/data/hybridrepre}}"
export TMPDIR="${HYBRIDREPRE_WORKDIR}/tmp"
export XDG_CACHE_HOME="${HYBRIDREPRE_WORKDIR}/cache"
export TORCH_HOME="${XDG_CACHE_HOME}/torch"
export HF_HOME="${XDG_CACHE_HOME}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TORCH_EXTENSIONS_DIR="${XDG_CACHE_HOME}/torch_extensions"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
export CUDA_CACHE_PATH="${XDG_CACHE_HOME}/cuda"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "${TMPDIR}" "${TORCH_HOME}" "${HF_HUB_CACHE}" "${TORCH_EXTENSIONS_DIR}" "${MPLCONFIGDIR}" "${CUDA_CACHE_PATH}"
