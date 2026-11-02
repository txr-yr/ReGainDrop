#!/usr/bin/env bash
# Modified for ReGainDrop: standalone paths, quoted arguments and input checks.
# Derived from the supplied VisPruner/LLaVA evaluation scripts; inference is unchanged.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/eval_common.sh"


# 模型目录 = ${CKPT_DIR}/${CKPT}，须指向含 config.json 的 LLaVA 权重根目录
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/models}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/datasets/sqa}"

CKPT="${CKPT:-llava-v1.5-7b}"
SPLIT="llava_test_CQM-I"



# 设为 1 时在推理结束后打印：加载时间、generate/e2e 延迟吞吐，以及 LLaVA1.5 TFLOPs 估算
SQA_BENCHMARK="${SQA_BENCHMARK:-1}"


# 与 answers-file 同目录：本项目 .../answers/${SPLIT}/${CKPT}/n_${TOKEN}/r_${RATIO}.jsonl

MODEL_PATH="${MODEL_PATH:-${CKPT_DIR}/${CKPT}}"
QUESTION_FILE="${QUESTION_FILE:-${REPO_ROOT}/playground/data/eval/scienceqa/${SPLIT}.json}"
require_file "${MODEL_PATH}/config.json"
require_file "${QUESTION_FILE}"
require_dir "${DATA_DIR}"
DATA_DIR="$(cd "${DATA_DIR}" && pwd)"
MODEL_PATH="$(cd "${MODEL_PATH}" && pwd)"
IMAGE_FOLDER="${IMAGE_FOLDER:-${DATA_DIR}/scienceqa/images/test}"
require_dir "${IMAGE_FOLDER}"

ANSWER_DIR="${REPO_ROOT}/playground/data/eval/scienceqa/answers/${SPLIT}/${CKPT}/n_${TOKEN}"
mkdir -p "${ANSWER_DIR}"

# Prune 可视化输出目录：
#   留空 — 不保存
#   answer 或 1 — 保存到 ANSWER_DIR/vis（答案目录下的 vis 子文件夹）
#   其它 — 任意绝对/相对路径
SQA_SAVE_PRUNE_VIS="${SQA_SAVE_PRUNE_VIS:-1}"
# 丢弃区域混合：out = 原图 * blend + 白 * (1-blend)，越小越淡
SQA_PRUNE_VIS_BLEND="${SQA_PRUNE_VIS_BLEND:-0.35}"

if [[ "${SQA_SAVE_PRUNE_VIS}" == "answer" ]] || [[ "${SQA_SAVE_PRUNE_VIS}" == "1" ]]; then
    SQA_SAVE_PRUNE_VIS="${ANSWER_DIR}/vis"
fi

BENCH_ARGS=()
if [[ "${SQA_BENCHMARK}" == "1" ]]; then
    BENCH_ARGS+=(--benchmark)
fi

PRUNE_ARGS=()
if [[ -n "${SQA_SAVE_PRUNE_VIS}" ]]; then
    PRUNE_ARGS+=(--save-prune-vis-dir "${SQA_SAVE_PRUNE_VIS}")
    PRUNE_ARGS+=(--prune-vis-discarded-blend "${SQA_PRUNE_VIS_BLEND}")
fi

# VisPruner 相似度×空间高斯核 sigma；不设则由 Python 默认 2.0
SPATIAL_SIGMA_ARGS=()
if [[ -n "${SQA_SPATIAL_SIGMA:-}" ]]; then
    SPATIAL_SIGMA_ARGS+=(--mm-vispruner-spatial-sigma "${SQA_SPATIAL_SIGMA}")
fi


python -m llava.eval.model_vqa_science \
    --model-path "${MODEL_PATH}" \
    --question-file "${QUESTION_FILE}" \
    --image-folder "${IMAGE_FOLDER}" \
    --answers-file "${ANSWER_DIR}/r_${RATIO}.jsonl" \
    --visual_token_num "${TOKEN}" \
    --important_ratio "${RATIO}" \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    "${BENCH_ARGS[@]}" \
    "${PRUNE_ARGS[@]}" \
    "${SPATIAL_SIGMA_ARGS[@]}"

python -m llava.eval.eval_science_qa \
    --base-dir "${DATA_DIR}/scienceqa" \
    --result-file "${ANSWER_DIR}/r_${RATIO}.jsonl" \
    --output-file "${ANSWER_DIR}/r_${RATIO}_output.jsonl" \
    --output-result "${ANSWER_DIR}/r_${RATIO}_result.json"
