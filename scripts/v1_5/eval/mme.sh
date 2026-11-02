#!/usr/bin/env bash
# Modified for ReGainDrop: standalone paths, quoted arguments and input checks.
# Derived from the supplied VisPruner/LLaVA evaluation scripts; inference is unchanged.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/eval_common.sh"

CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/models}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/datasets}"

CKPT="${CKPT:-llava-v1.5-7b}"
SPLIT="llava_mme"



MODEL_PATH="${MODEL_PATH:-${CKPT_DIR}/${CKPT}}"
QUESTION_FILE="${QUESTION_FILE:-./playground/data/eval/MME/${SPLIT}.jsonl}"
require_file "${MODEL_PATH}/config.json"
require_file "${QUESTION_FILE}"
require_dir "${DATA_DIR}"
DATA_DIR="$(cd "${DATA_DIR}" && pwd)"
MODEL_PATH="$(cd "${MODEL_PATH}" && pwd)"
IMAGE_FOLDER="${IMAGE_FOLDER:-${DATA_DIR}/MME/MME_Benchmark_release_version}"
require_dir "${IMAGE_FOLDER}"


python -m llava.eval.model_vqa_loader \
    --model-path "${MODEL_PATH}" \
    --question-file "${QUESTION_FILE}" \
    --image-folder "${IMAGE_FOLDER}" \
    --answers-file "./playground/data/eval/MME/answers/${SPLIT}/${CKPT}/n_${TOKEN}/r_${RATIO}.jsonl" \
    --visual_token_num "${TOKEN}" \
    --important_ratio "${RATIO}" \
    --temperature 0 \
    --conv-mode vicuna_v1

cd "./playground/data/eval/MME"

python convert_answer_to_mme.py \
    --data_path "${DATA_DIR}/MME" \
    --experiment "${SPLIT}/${CKPT}/n_${TOKEN}/r_${RATIO}"

cd eval_tool

python calculation.py --results_dir answers/${SPLIT}/${CKPT}/n_${TOKEN}/r_${RATIO}
