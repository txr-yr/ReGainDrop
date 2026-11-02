#!/usr/bin/env bash
# Modified for ReGainDrop: standalone paths, quoted arguments and input checks.
# Derived from the supplied VisPruner/LLaVA evaluation scripts; inference is unchanged.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/eval_common.sh"


CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/models}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/datasets/pope}"

CKPT="${CKPT:-llava-v1.6-vicuna-7b}"
SPLIT="llava_pope_test"

POPE_BENCHMARK="${POPE_BENCHMARK:-1}"



MODEL_PATH="${MODEL_PATH:-${CKPT_DIR}/${CKPT}}"
QUESTION_FILE="${QUESTION_FILE:-./playground/data/eval/pope/${SPLIT}.jsonl}"
require_file "${MODEL_PATH}/config.json"
require_file "${QUESTION_FILE}"
require_dir "${DATA_DIR}"
DATA_DIR="$(cd "${DATA_DIR}" && pwd)"
MODEL_PATH="$(cd "${MODEL_PATH}" && pwd)"
IMAGE_FOLDER="${IMAGE_FOLDER:-${DATA_DIR}/pope/val2014}"
require_dir "${IMAGE_FOLDER}"

ANSWER_DIR="${REPO_ROOT}/playground/data/eval/pope/answers/${SPLIT}/${CKPT}/n_${TOKEN}"

POPE_SAVE_PRUNE_VIS="${POPE_SAVE_PRUNE_VIS:-1}"
POPE_PRUNE_VIS_BLEND="${POPE_PRUNE_VIS_BLEND:-0.35}"

if [[ "${POPE_SAVE_PRUNE_VIS}" == "answer" ]] || [[ "${POPE_SAVE_PRUNE_VIS}" == "1" ]]; then
    POPE_SAVE_PRUNE_VIS="${ANSWER_DIR}/vis"
fi

BENCH_ARGS=()
if [[ "${POPE_BENCHMARK}" == "1" ]]; then
    BENCH_ARGS+=(--benchmark)
fi

PRUNE_ARGS=()
if [[ -n "${POPE_SAVE_PRUNE_VIS}" ]]; then
    PRUNE_ARGS+=(--save-prune-vis-dir "${POPE_SAVE_PRUNE_VIS}")
    PRUNE_ARGS+=(--prune-vis-discarded-blend "${POPE_PRUNE_VIS_BLEND}")
fi


python -m llava.eval.model_vqa_loader \
    --model-path "${MODEL_PATH}" \
    --question-file "${QUESTION_FILE}" \
    --image-folder "${IMAGE_FOLDER}" \
    --answers-file "./playground/data/eval/pope/answers/${SPLIT}/${CKPT}/n_${TOKEN}/r_${RATIO}.jsonl" \
    --visual_token_num "${TOKEN}" \
    --important_ratio "${RATIO}" \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    "${BENCH_ARGS[@]}" \
    "${PRUNE_ARGS[@]}"

python llava/eval/eval_pope.py \
    --annotation-dir "${DATA_DIR}/pope/coco" \
    --question-file "${QUESTION_FILE}" \
    --result-file "./playground/data/eval/pope/answers/${SPLIT}/${CKPT}/n_${TOKEN}/r_${RATIO}.jsonl"
