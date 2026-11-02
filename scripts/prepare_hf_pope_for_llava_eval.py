# ReGainDrop release preparation (2026-11-02): distribution notice added.
# Implementation retained from the author-supplied development snapshot; see NOTICE.
# Modified for ReGainDrop release: portable default repository location.
"""
从 Hugging Face lmms-lab/POPE 风格 parquet 生成 VisPruner pope.sh 所需数据：

  {out_data}/pope/val2014/COCO_val2014_*.jpg
  {out_data}/pope/coco/coco_pope_{popular,random,adversarial}.json  （每行一个 JSON，含 label）

  {repo}/playground/data/eval/pope/llava_pope_test.jsonl

jsonl 每行字段：question_id（全局唯一）、image（相对 val2014）、text、category

说明：HF 各 split 内 question_id 均为 1..3000，故全局 id 使用 \"{category}_{local_id}\"。
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pandas as pd
from PIL import Image

_CATEGORY_ORDER = ("popular", "random", "adversarial")


def _load_parquet_tables(data_root: Path, hf_config: str):
    if hf_config == "default":
        sub = data_root / "data"
        files = sorted(sub.glob("test-*.parquet"))
        if not files:
            raise FileNotFoundError(f"{sub} 下无 test-*.parquet")
        dfs = [pd.read_parquet(f) for f in files]
        return pd.concat(dfs, ignore_index=True)
    if hf_config == "Full":
        dfs = []
        for split in _CATEGORY_ORDER:
            files = sorted((data_root / "Full").glob(f"{split}-*.parquet"))
            if not files:
                raise FileNotFoundError(f"缺少 Full/{split}-*.parquet")
            dfs.append(pd.read_parquet(files[0]))
        return pd.concat(dfs, ignore_index=True)
    raise ValueError(f"未知 --hf-config: {hf_config}")


def _sort_for_llava(df: pd.DataFrame) -> pd.DataFrame:
    """与 coco_pope_*.json 块顺序一致：popular -> random -> adversarial，块内按本地 question_id。"""
    cat_rank = {c: i for i, c in enumerate(_CATEGORY_ORDER)}
    df = df.copy()
    df["_cat_rank"] = df["category"].map(cat_rank)
    df["_qid"] = df["question_id"].astype(int)
    return df.sort_values(["_cat_rank", "_qid"]).drop(columns=["_cat_rank", "_qid"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="./datasets/pope")
    parser.add_argument(
        "--hf-config",
        type=str,
        default="default",
        choices=["default", "Full"],
        help="default: data/test-*.parquet；Full: Full/{popular,random,adversarial}-*.parquet",
    )
    parser.add_argument(
        "--out-data-root",
        type=str,
        default=None,
        help="写出 pope/val2014 与 pope/coco 的根目录，默认与 --data-root 相同",
    )
    parser.add_argument("--repo-root", type=str, default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument(
        "--text-suffix",
        type=str,
        default=" Please answer yes or no.",
        help="拼在 question 后的提示；设为空字符串则仅原始 question",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    out_root = Path(args.out_data_root).resolve() if args.out_data_root else data_root
    repo_root = Path(args.repo_root).resolve()

    df = _load_parquet_tables(data_root, args.hf_config)
    df = _sort_for_llava(df)

    pope_dir = out_root / "pope"
    img_dir = pope_dir / "val2014"
    coco_dir = pope_dir / "coco"
    img_dir.mkdir(parents=True, exist_ok=True)
    coco_dir.mkdir(parents=True, exist_ok=True)

    jsonl_lines = []
    labels_by_cat: dict[str, list[str]] = {c: [] for c in _CATEGORY_ORDER}

    for _, row in df.iterrows():
        cat = str(row["category"])
        if cat not in labels_by_cat:
            raise ValueError(f"未知 category: {cat}")
        local_id = int(row["question_id"])
        qid_global = f"{cat}_{local_id}"

        stem = str(row["image_source"]).strip()
        if not stem.lower().endswith(".jpg"):
            fname = f"{stem}.jpg"
        else:
            fname = stem

        cell = row["image"]
        if not isinstance(cell, dict) or not cell.get("bytes"):
            raise ValueError(f"question_id={qid_global} 无 image bytes")
        im = Image.open(io.BytesIO(cell["bytes"])).convert("RGB")
        im.save(img_dir / fname)

        question = str(row["question"]).strip()
        text = question + (args.text_suffix or "")

        jsonl_lines.append(
            {
                "question_id": qid_global,
                "image": fname,
                "text": text,
                "category": cat,
            }
        )

        ans = str(row["answer"]).strip().lower()
        if ans not in ("yes", "no"):
            raise ValueError(f"answer 应为 yes/no，得到: {row['answer']}")
        labels_by_cat[cat].append(ans)

    eval_pope = repo_root / "playground" / "data" / "eval" / "pope"
    eval_pope.mkdir(parents=True, exist_ok=True)
    jsonl_path = eval_pope / "llava_pope_test.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for obj in jsonl_lines:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    for cat in _CATEGORY_ORDER:
        labels = labels_by_cat[cat]
        ann_path = coco_dir / f"coco_pope_{cat}.json"
        with open(ann_path, "w", encoding="utf-8") as f:
            for lab in labels:
                f.write(json.dumps({"label": lab}, ensure_ascii=False) + "\n")

    print(f"jsonl: {len(jsonl_lines)} 条 -> {jsonl_path}", flush=True)
    print(f"图片 -> {img_dir}", flush=True)
    for cat in _CATEGORY_ORDER:
        print(f"标注 {cat}: {len(labels_by_cat[cat])} 行 -> {coco_dir / f'coco_pope_{cat}.json'}", flush=True)


if __name__ == "__main__":
    main()
