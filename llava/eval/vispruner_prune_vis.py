# ReGainDrop release preparation (2026-11-02): distribution notice added.
# Implementation retained from the author-supplied development snapshot; see NOTICE.
"""将 VisPruner 丢弃的视觉 patch 在原图对应区域做淡化（模拟更低不透明度/更淡）并保存。"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from llava.mm_utils import expand2square


def _crop_square_to_original(square: Image.Image, orig_size: tuple[int, int]) -> Image.Image:
    w, h = orig_size
    S = square.size[0]
    assert square.size[1] == S
    if w >= h:
        y0 = (S - h) // 2
        return square.crop((0, y0, w, y0 + h))
    x0 = (S - w) // 2
    return square.crop((x0, 0, x0 + w, h))


def render_discarded_patches_on_original(
    original_pil: Image.Image,
    index_mask_1d: torch.Tensor,
    num_patches_per_side: int,
    image_processor,
    discarded_blend: float = 0.35,
) -> Image.Image:
    """
    index_mask_1d: bool [N]，与 encode_images 一致，True=保留，False=丢弃。
    在 expand2square 后与 CLIP 输入 224 对齐的缩放网格上淡化丢弃块，再裁回原始分辨率。

    discarded_blend: 丢弃区域 out = orig * blend + 255 * (1 - blend)，越小越“透明/淡”。
    """
    if discarded_blend < 0 or discarded_blend > 1:
        raise ValueError("discarded_blend 应在 [0,1]")

    original_pil = original_pil.convert("RGB")
    ow, oh = original_pil.size
    bg = tuple(int(x * 255) for x in image_processor.image_mean)
    sq = expand2square(original_pil, bg)
    S = sq.size[0]

    H = W = num_patches_per_side
    n = H * W
    mask = index_mask_1d.reshape(-1).bool().cpu()
    if mask.numel() != n:
        raise ValueError(
            f"index_mask 长度 {mask.numel()} 与网格 {n}={H}x{W} 不一致（anyres 等布局请关闭可视化或扩展实现）"
        )

    target = int(image_processor.crop_size["height"])
    p_pix = target // H
    if p_pix * H != target:
        raise ValueError(f"crop_size {target} 无法整除 patch 边长 {H}")

    scale = S / float(target)
    arr = np.array(sq, dtype=np.float32)

    for idx in range(n):
        if mask[idx].item():
            continue
        pr, pc = divmod(idx, W)
        x0 = int(round(pc * p_pix * scale))
        x1 = int(round((pc + 1) * p_pix * scale))
        y0 = int(round(pr * p_pix * scale))
        y1 = int(round((pr + 1) * p_pix * scale))
        x0 = max(0, min(x0, S - 1))
        y0 = max(0, min(y0, S - 1))
        x1 = max(x0 + 1, min(x1, S))
        y1 = max(y0 + 1, min(y1, S))
        patch = arr[y0:y1, x0:x1]
        patch[:] = patch * discarded_blend + 255.0 * (1.0 - discarded_blend)

    out_sq = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return _crop_square_to_original(out_sq, (ow, oh))
