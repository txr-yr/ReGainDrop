#!/usr/bin/env bash
# Modified for ReGainDrop: standalone paths, quoted arguments and input checks.
# Derived from the supplied VisPruner/LLaVA evaluation scripts; inference is unchanged.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/eval_common.sh"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/models}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/datasets}"

CKPT="${CKPT:-llava-v1.6-vicuna-7b}"
SPLIT="llava_gqa_testdev_balanced"



MODEL_PATH="${MODEL_PATH:-${CKPT_DIR}/${CKPT}}"
QUESTION_FILE="${QUESTION_FILE:-./playground/data/eval/gqa/${SPLIT}.jsonl}"
require_file "${MODEL_PATH}/config.json"
require_file "${QUESTION_FILE}"
require_dir "${DATA_DIR}"
DATA_DIR="$(cd "${DATA_DIR}" && pwd)"
MODEL_PATH="$(cd "${MODEL_PATH}" && pwd)"
IMAGE_FOLDER="${IMAGE_FOLDER:-${DATA_DIR}/gqa/data/images}"
require_dir "${IMAGE_FOLDER}"

PIDS=()
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES="${GPULIST[$IDX]}" python -m llava.eval.model_vqa_loader \
        --model-path "${MODEL_PATH}" \
        --question-file "${QUESTION_FILE}" \
        --image-folder "${IMAGE_FOLDER}" \
        --answers-file "./playground/data/eval/gqa/answers/${SPLIT}/${CKPT}/n_${TOKEN}/r_${RATIO}/${CHUNKS}_${IDX}.jsonl" \
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

GQA_DIR="./playground/data/eval/gqa/data"
output_file="./playground/data/eval/gqa/answers/${SPLIT}/${CKPT}/n_${TOKEN}/r_${RATIO}/merge.jsonl"

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat "./playground/data/eval/gqa/answers/${SPLIT}/${CKPT}/n_${TOKEN}/r_${RATIO}/${CHUNKS}_${IDX}.jsonl" >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src "$output_file" --dst "${GQA_DIR}/testdev_balanced_predictions.json"

cd "${GQA_DIR}"
python eval/eval.py \
    --path "${DATA_DIR}/gqa/data/questions" \
    --tier testdev_balanced
