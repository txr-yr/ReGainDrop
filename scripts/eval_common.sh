#!/usr/bin/env bash
# ReGainDrop release modification: portable paths and early input validation.
# Evaluation/model code retains its upstream attribution and checkpoint API.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
if [[ $# -ne 2 ]]; then
    printf 'Usage: bash %s VISUAL_TOKEN_NUMBER IMPORTANT_RATIO\n' "$0" >&2
    exit 2
fi
TOKEN="$1"
RATIO="$2"
if [[ ! "$TOKEN" =~ ^[1-9][0-9]*$ ]]; then
    printf 'VISUAL_TOKEN_NUMBER must be a positive integer.\n' >&2
    exit 2
fi
if [[ ! "$RATIO" =~ ^(0([.][0-9]+)?|1([.]0+)?)$ ]]; then
    printf 'IMPORTANT_RATIO must be between 0 and 1 (for example 0.5).\n' >&2
    exit 2
fi
command -v python >/dev/null || { printf 'Activate the LLaVA Python environment first.\n' >&2; exit 2; }
require_file() {
    [[ -f "$1" ]] || { printf 'Required file is missing: %s\n' "$1" >&2; exit 2; }
}
require_dir() {
    [[ -d "$1" ]] || { printf 'Required directory is missing: %s\n' "$1" >&2; exit 2; }
}
