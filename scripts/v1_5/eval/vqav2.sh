#!/usr/bin/env bash
# Modified for ReGainDrop: standalone paths, quoted arguments and input checks.
# Derived from the supplied VisPruner/LLaVA evaluation scripts; inference is unchanged.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/eval_common.sh"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/models}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/datasets}"

CKPT="${CKPT:-llava-v1.5-7b}"
SPLIT="llava_vqav2_mscoco_test-dev2015"



MODEL_PATH="${MODEL_PATH:-${CKPT_DIR}/${CKPT}}"
QUESTION_FILE="${QUESTION_FILE:-./playground/data/eval/vqav2/${SPLIT}.jsonl}"
require_file "${MODEL_PATH}/config.json"
require_file "${QUESTION_FILE}"
require_dir "${DATA_DIR}"
DATA_DIR="$(cd "${DATA_DIR}" && pwd)"
MODEL_PATH="$(cd "${MODEL_PATH}" && pwd)"
IMAGE_FOLDER="${IMAGE_FOLDER:-${DATA_DIR}/vqav2/test2015}"
require_dir "${IMAGE_FOLDER}"

PIDS=()
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES="${GPULIST[$IDX]}" python -m llava.eval.model_vqa_loader \
        --model-path "${MODEL_PATH}" \
        --question-file "${QUESTION_FILE}" \
        --image-folder "${IMAGE_FOLDER}" \
        --answers-file "./playground/data/eval/vqav2/answers/${SPLIT}/${CKPT}/n_${TOKEN}/r_${RATIO}/${CHUNKS}_${IDX}.jsonl" \
        --num-chunks "${CHUNKS}" \
        --chunk-idx "${IDX}" \
        --visual_token_num "${TOKEN}" \
        --important_ratio "${RATIO}" \
        --temperature 0 \
        --conv-mode vicuna_v1 &
    PIDS+=("$!")
done

for worker_pid in "${PIDS[@]}"; do
    wait "$worker_pid"
done

VQAV2_DIR="./playground/data/eval/vqav2"
output_file="./playground/data/eval/vqav2/answers/${SPLIT}/${CKPT}/n_${TOKEN}/r_${RATIO}/merge.jsonl"

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat "./playground/data/eval/vqav2/answers/${SPLIT}/${CKPT}/n_${TOKEN}/r_${RATIO}/${CHUNKS}_${IDX}.jsonl" >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py \
    --dir "${VQAV2_DIR}" \
    --src "answers/${SPLIT}/${CKPT}/n_${TOKEN}/r_${RATIO}/merge.jsonl" \
    --dst "answers_upload/${SPLIT}/${CKPT}/n_${TOKEN}/r_${RATIO}.json"
