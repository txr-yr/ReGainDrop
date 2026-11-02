#!/usr/bin/env bash
# Modified for ReGainDrop: standalone paths, quoted arguments and input checks.
# Derived from the supplied VisPruner/LLaVA evaluation scripts; inference is unchanged.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/eval_common.sh"

CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/models}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/datasets}"

CKPT="${CKPT:-llava-v1.5-7b}"
SPLIT="mmbench_dev_20230712"



MODEL_PATH="${MODEL_PATH:-${CKPT_DIR}/${CKPT}}"
QUESTION_FILE="${QUESTION_FILE:-${DATA_DIR}/mmbench/${SPLIT}.tsv}"
require_file "${MODEL_PATH}/config.json"
require_file "${QUESTION_FILE}"
require_dir "${DATA_DIR}"
DATA_DIR="$(cd "${DATA_DIR}" && pwd)"
MODEL_PATH="$(cd "${MODEL_PATH}" && pwd)"


python -m llava.eval.model_vqa_mmbench \
    --model-path "${MODEL_PATH}" \
    --question-file "${QUESTION_FILE}" \
    --answers-file "./playground/data/eval/mmbench/answers/${SPLIT}/${CKPT}/n_${TOKEN}/r_${RATIO}.jsonl" \
    --visual_token_num "${TOKEN}" \
    --important_ratio "${RATIO}" \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p "playground/data/eval/mmbench/answers_upload/${SPLIT}/${CKPT}/n_${TOKEN}"

python scripts/convert_mmbench_for_submission.py \
    --annotation-file "${DATA_DIR}/mmbench/${SPLIT}.tsv" \
    --result-dir "./playground/data/eval/mmbench/answers/${SPLIT}/${CKPT}/n_${TOKEN}" \
    --upload-dir "./playground/data/eval/mmbench/answers_upload/${SPLIT}/${CKPT}/n_${TOKEN}" \
    --experiment "r_${RATIO}"
