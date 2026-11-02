# ReGainDrop release preparation (2026-11-02): distribution notice added.
# Implementation retained from the author-supplied development snapshot; see NOTICE.
# Modified for ReGainDrop release: portable default repository location.
"""
从 lmms-lab/ScienceQA（parquet）生成 LLaVA/VisPruner sqa.sh 所需目录结构：

  {out_root}/scienceqa/problems.json
  {out_root}/scienceqa/pid_splits.json
  {out_root}/scienceqa/images/test/{problem_id}/image.png

并生成问题文件：

  playground/data/eval/scienceqa/llava_test_CQM-I.json

默认使用 ScienceQA-FULL 的 test 划分（4241 条，含纯文本 + 多模态）。
使用 --hf-config ScienceQA-IMG 则仅用带图子集 test（约 2017 条，与 sqa.sh 路径不变时可覆盖同名 json）。
"""
import argparse
import io
import json
import os
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

# scripts/ 在 path 中时可导入
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from convert_sqa_to_llava_base_prompt import build_prompt_chatbot  # noqa: E402


def row_to_problem(row) -> dict:
    choices = row["choices"]
    if hasattr(choices, "tolist"):
        choices = choices.tolist()
    choices = [str(c) for c in choices]
    hint = row["hint"] if pd.notna(row["hint"]) else ""
    lecture = row["lecture"] if pd.notna(row["lecture"]) else ""
    solution = row["solution"] if pd.notna(row["solution"]) else ""
    answer = int(row["answer"])
    image = None
    if pd.notna(row["image"]) and row["image"] is not None:
        image = "image.png"
    return {
        "question": str(row["question"]),
        "hint": hint,
        "choices": choices,
        "answer": answer,
        "lecture": lecture,
        "solution": solution,
        "image": image,
        "caption": "",
    }


def save_image_cell(cell, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(cell, dict) and cell.get("bytes"):
        im = Image.open(io.BytesIO(cell["bytes"]))
        im.save(path)
    else:
        raise ValueError("无法解析图像字段（需要含 bytes 的 HF image dict）")


def _load_test_parquet(data_root: Path, hf_config: str):
    cfg_dir = data_root / hf_config
    if not cfg_dir.is_dir():
        raise FileNotFoundError(f"找不到子集目录: {cfg_dir}")
    files = sorted(cfg_dir.glob("test-*.parquet"))
    if not files:
        raise FileNotFoundError(f"{cfg_dir} 下没有 test-*.parquet")
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet",
        type=str,
        default=None,
        help="直接指定 test parquet 文件；若设置则忽略 --hf-config",
    )
    parser.add_argument(
        "--hf-config",
        type=str,
        default="ScienceQA-FULL",
        choices=["ScienceQA-FULL", "ScienceQA-IMG"],
        help="使用 datasets/sqa 下哪个子目录中的 test-*.parquet（ScienceQA-IMG 仅多模态题）",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./datasets/sqa",
        help="含 ScienceQA-FULL / ScienceQA-IMG 等子目录的数据根目录",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default=None,
        help="输出 scienceqa 子目录的父路径，默认同 --data-root",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=".",
        help="项目根目录（用于写 playground/...）",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default="test",
        help="写入 pid_splits.json 的划分名（eval_science_qa 默认读 test）",
    )
    parser.add_argument(
        "--prompt-format",
        type=str,
        default="CQM-I",
        help="与 sqa.sh 中 SPLIT=llava_{split}_{format} 对应",
    )
    parser.add_argument(
        "--id-prefix",
        type=str,
        default="",
        help="问题 ID 前缀，默认使用数字字符串 0..N-1",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    out_root = Path(args.out_root).resolve() if args.out_root else data_root
    repo_root = Path(args.repo_root).resolve()

    if args.parquet:
        pq_path = Path(args.parquet).resolve()
        if not pq_path.is_file():
            raise FileNotFoundError(f"找不到 parquet: {pq_path}")
        df = pd.read_parquet(pq_path)
    else:
        df = _load_test_parquet(data_root, args.hf_config)

    if args.hf_config == "ScienceQA-IMG" and not args.parquet:
        bad = df["image"].isna() | (df["image"].apply(lambda x: x is None))
        n_bad = int(bad.sum())
        if n_bad:
            raise ValueError(
                f"ScienceQA-IMG 期望每行均有图，当前有 {n_bad} 行 image 为空，请检查数据"
            )
    scienceqa_dir = out_root / "scienceqa"
    img_root = scienceqa_dir / "images" / args.split_name
    scienceqa_dir.mkdir(parents=True, exist_ok=True)

    problems = {}
    split_ids = []

    for i in range(len(df)):
        pid = f"{args.id_prefix}{i}"
        row = df.iloc[i]
        prob = row_to_problem(row)
        problems[pid] = prob
        split_ids.append(pid)

        if prob["image"] is not None:
            cell = row["image"]
            save_image_cell(cell, img_root / pid / "image.png")

    pid_splits = {
        args.split_name: split_ids,
        "train": [],
        "validation": [],
    }
    if args.split_name not in ("train", "validation"):
        pass
    with open(scienceqa_dir / "problems.json", "w", encoding="utf-8") as f:
        json.dump(problems, f, ensure_ascii=False)
    with open(scienceqa_dir / "pid_splits.json", "w", encoding="utf-8") as f:
        json.dump(pid_splits, f, ensure_ascii=False)

    split_problems = build_prompt_chatbot(
        problems,
        split_ids,
        args.prompt_format,
        use_caption=False,
        is_test=True,
    )

    target_format = []
    for prob_id in split_ids:
        input_text, output_text = split_problems[prob_id]
        if input_text.startswith("Question: "):
            input_text = input_text.replace("Question: ", "", 1)
        if output_text.startswith("Answer: "):
            output_text = output_text.replace("Answer: ", "", 1)

        raw = problems[prob_id]
        if raw["image"] is None:
            target_format.append(
                {
                    "id": prob_id,
                    "conversations": [
                        {"from": "human", "value": f"{input_text}"},
                        {"from": "gpt", "value": f"{output_text}"},
                    ],
                }
            )
        else:
            rel_img = os.path.join(prob_id, raw["image"])
            target_format.append(
                {
                    "id": prob_id,
                    "image": rel_img,
                    "conversations": [
                        {"from": "human", "value": f"{input_text}\n<image>"},
                        {"from": "gpt", "value": f"{output_text}"},
                    ],
                }
            )

    eval_dir = repo_root / "playground" / "data" / "eval" / "scienceqa"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"llava_{args.split_name}_{args.prompt_format}.json"
    out_json = eval_dir / out_name
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(target_format, f, ensure_ascii=False, indent=2)

    print(
        f"已写入 {len(target_format)} 条（hf-config={args.hf_config}）-> {out_json}",
        flush=True,
    )
    print(f"标注与图片 -> {scienceqa_dir}", flush=True)


if __name__ == "__main__":
    main()
