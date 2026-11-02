#!/usr/bin/env bash
# Modified for ReGainDrop: standalone paths, quoted arguments and input checks.
# Derived from the supplied VisPruner/LLaVA evaluation scripts; inference is unchanged.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/eval_common.sh"

CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/models}"
# 转换后目录结构：${DATA_DIR}/pope/val2014、${DATA_DIR}/pope/coco（见 prepare_hf_pope_for_llava_eval.py）
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/datasets/pope}"

CKPT="${CKPT:-llava-v1.5-7b}"
SPLIT="llava_pope_test"

# POPE 子集：all（9000 条）；当前默认 popular| popular | random | adversarial | 逗号组合如 popular,random
# 非 all 时结果写入 n_${TOKEN}/<子目录>/r_${RATIO}.jsonl，避免与全量互相覆盖
POPE_CATEGORY="${POPE_CATEGORY:-popular}"

# 设为 1 时打印：load_time(s)、generate/e2e 延迟吞吐，以及 LLaVA1.5 TFLOPs 估算
POPE_BENCHMARK="${POPE_BENCHMARK:-1}"





MODEL_PATH="${MODEL_PATH:-${CKPT_DIR}/${CKPT}}"
QUESTION_FILE="${QUESTION_FILE:-${REPO_ROOT}/playground/data/eval/pope/${SPLIT}.jsonl}"
require_file "${MODEL_PATH}/config.json"
require_file "${QUESTION_FILE}"
require_dir "${DATA_DIR}"
DATA_DIR="$(cd "${DATA_DIR}" && pwd)"
MODEL_PATH="$(cd "${MODEL_PATH}" && pwd)"
IMAGE_FOLDER="${IMAGE_FOLDER:-${DATA_DIR}/pope/val2014}"
require_dir "${IMAGE_FOLDER}"

ANS_N="playground/data/eval/pope/answers/${SPLIT}/${CKPT}/n_${TOKEN}"
POPE_CAT_ARGS=()
EVAL_CAT_ARGS=()

if [[ "${POPE_CATEGORY}" == "all" ]] || [[ -z "${POPE_CATEGORY}" ]]; then
  ANS_REL="./${ANS_N}/r_${RATIO}.jsonl"
  ANSWER_DIR="${REPO_ROOT}/${ANS_N}"
else
  # 子目录名：类别名排序后用下划线连接，如 adversarial_popular_random
  CAT_SUBDIR="$(echo "${POPE_CATEGORY}" | tr ',' '\n' | tr -d ' ' | sort -u | paste -sd '_')"
  ANS_REL="./${ANS_N}/${CAT_SUBDIR}/r_${RATIO}.jsonl"
  ANSWER_DIR="${REPO_ROOT}/${ANS_N}/${CAT_SUBDIR}"
  POPE_CAT_ARGS+=(--pope-categories "${POPE_CATEGORY}")
  EVAL_CAT_ARGS+=(--categories "${POPE_CATEGORY}")
fi

mkdir -p "${ANSWER_DIR}"

# Prune 可视化：留空不保存；answer/1 -> ANSWER_DIR/vis；其它为自定义路径
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

# VisPruner 相似度×空间高斯核 sigma（默认 2.0；可在运行前 export POPE_SPATIAL_SIGMA=0.3 覆盖）
SPATIAL_SIGMA_ARGS=()
if [[ -n "${POPE_SPATIAL_SIGMA:-}" ]]; then
  SPATIAL_SIGMA_ARGS+=(--mm-vispruner-spatial-sigma "${POPE_SPATIAL_SIGMA}")
fi

# ViT 热力图：在「指定层」上画 CLS 注意力 与 EUTI-VTT（与剪枝用的 mm_vision_select_layer 可不同；需 matplotlib）
# POPE_VIT_HEATMAP_LAYER：encoder 下标，0=第一层，-1=最后一层；留空则优先用 checkpoint 的 mm_vision_heatmap_layer，再否则用 mm_vision_select_layer
# 设 POPE_SAVE_VIT_HEATMAP=0 可关闭
POPE_SAVE_VIT_HEATMAP="${POPE_SAVE_VIT_HEATMAP:-0}"
POPE_VIT_HEATMAP_LAYER="${POPE_VIT_HEATMAP_LAYER:-20}"
VIT_HM_ARGS=()
if [[ "${POPE_SAVE_VIT_HEATMAP}" != "0" ]] && [[ -z "${POPE_SKIP_VIT_HEATMAP:-}" ]]; then
  POPE_VIT_HEATMAP_CLS_DIR="${POPE_VIT_HEATMAP_CLS_DIR:-${ANSWER_DIR}/vit_heatmap_cls}"
  POPE_VIT_HEATMAP_VTT_DIR="${POPE_VIT_HEATMAP_VTT_DIR:-${ANSWER_DIR}/vit_heatmap_vtt}"
  mkdir -p "${POPE_VIT_HEATMAP_CLS_DIR}" "${POPE_VIT_HEATMAP_VTT_DIR}"
  VIT_HM_ARGS+=(--save-vit-heatmap-cls-dir "${POPE_VIT_HEATMAP_CLS_DIR}")
  VIT_HM_ARGS+=(--save-vit-heatmap-vtt-dir "${POPE_VIT_HEATMAP_VTT_DIR}")
  if [[ -n "${POPE_VIT_HEATMAP_LAYER}" ]]; then
    VIT_HM_ARGS+=(--save-vit-heatmap-layer "${POPE_VIT_HEATMAP_LAYER}")
  fi
fi


python -m llava.eval.model_vqa_loader \
  --model-path "${MODEL_PATH}" \
  --question-file "${QUESTION_FILE}" \
  --image-folder "${IMAGE_FOLDER}" \
  --answers-file "${ANS_REL}" \
  --visual_token_num "${TOKEN}" \
  --important_ratio "${RATIO}" \
  --temperature 0 \
  --conv-mode vicuna_v1 \
  "${BENCH_ARGS[@]}" \
  "${PRUNE_ARGS[@]}" \
  "${VIT_HM_ARGS[@]}" \
  "${POPE_CAT_ARGS[@]}" \
  "${SPATIAL_SIGMA_ARGS[@]}"

python llava/eval/eval_pope.py \
  --annotation-dir "${DATA_DIR}/pope/coco" \
  --question-file "${QUESTION_FILE}" \
  --result-file "${ANS_REL}" \
  "${EVAL_CAT_ARGS[@]}"
