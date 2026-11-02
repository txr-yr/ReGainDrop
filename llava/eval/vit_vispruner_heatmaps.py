# ReGainDrop release preparation (2026-11-02): distribution notice added.
# Implementation retained from the author-supplied development snapshot; see NOTICE.
"""VisPruner：将 ViT 选定层上每个 patch 的 CLS 注意力与 VTT(EUTI) 分数可视化为平方热力图 PNG。"""

import os
from typing import Optional, Sequence, Union

import numpy as np
import torch

try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None
    matplotlib = None


def _vec_to_square(
    vec: torch.Tensor,
    num_patches_per_side: int,
) -> np.ndarray:
    """(N,) -> (H, W)，N 须等于 H*W。"""
    n = int(vec.numel())
    s = int(num_patches_per_side)
    if n != s * s:
        raise ValueError(f"patch 数 {n} 与 grid {s}x{s} 不一致")
    x = vec.detach().float().cpu().numpy().reshape(s, s)
    return x


def _normalize_for_green_heatmap(arr: np.ndarray) -> np.ndarray:
    """
    分位数拉伸避免极值拉满对比；再映射到 Greens colormap 中段，减小色相跳跃。
    """
    lo, hi = np.percentile(arr, [10.0, 90.0])
    if hi - lo < 1e-12:
        return np.full_like(arr, 0.52, dtype=np.float64)
    t = (arr - lo) / (hi - lo)
    t = np.clip(t, 0.0, 1.0)
    # 略压低高亮差：略 compress
    t = np.power(t, 0.92)
    # 仅使用 Greens 中间一段 [vmin, vmax]，避免最浅/最深两端变化过猛
    cmin, cmax = 0.38, 0.82
    return cmin + (cmax - cmin) * t


def save_vit_prune_heatmaps_batch(
    cls_scores: torch.Tensor,
    vtt_scores: torch.Tensor,
    num_patches_per_side: int,
    cls_dir: Optional[str],
    vtt_dir: Optional[str],
    qids: Union[str, Sequence[str]],
    *,
    cmap: str = "Greens",
) -> None:
    """
    cls_scores / vtt_scores: (B, N) 或 (N,) 的逐 patch 标量。
    qids: 单字符串或长度 B 的 question_id 列表（用于文件名）。
    输出分辨率为 patch 网格（num_patches_per_side × num_patches_per_side）像素。
    """
    if cls_dir is None and vtt_dir is None:
        return
    if plt is None or matplotlib is None:
        raise ImportError("保存热力图需要 matplotlib，请 pip install matplotlib")

    if cls_scores.dim() == 1:
        cls_scores = cls_scores.unsqueeze(0)
        vtt_scores = vtt_scores.unsqueeze(0)
    bsz = cls_scores.shape[0]
    if isinstance(qids, (str, int)):
        qids_list = [str(qids)] * bsz
    else:
        qids_list = [str(x).replace("/", "_") for x in qids]
        if len(qids_list) < bsz:
            qids_list = qids_list + [f"batch_{i}" for i in range(len(qids_list), bsz)]

    for bi in range(bsz):
        qid = qids_list[bi]
        for scores, out_dir, tag in (
            (cls_scores[bi], cls_dir, "cls"),
            (vtt_scores[bi], vtt_dir, "vtt"),
        ):
            if out_dir is None:
                continue
            os.makedirs(out_dir, exist_ok=True)
            arr = _vec_to_square(scores, num_patches_per_side)
            norm = _normalize_for_green_heatmap(arr)
            try:
                cmap_obj = matplotlib.colormaps[cmap]
            except (AttributeError, KeyError):
                cmap_obj = plt.get_cmap(cmap)
            rgba = cmap_obj(norm)
            rgb = (rgba[..., :3] * 255.0).clip(0, 255).astype(np.uint8)
            try:
                from PIL import Image

                Image.fromarray(rgb).save(os.path.join(out_dir, f"{qid}.png"))
            except Exception as e:  # pragma: no cover
                print(f"[vit_vispruner_heatmaps] 保存失败 ({tag}, qid={qid}): {e}", flush=True)
