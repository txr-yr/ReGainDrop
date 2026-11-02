# ReGainDrop release preparation (2026-11-02): distribution notice added.
# Implementation retained from the author-supplied development snapshot; see NOTICE.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPVisionModel, CLIPVisionConfig, CLIPImageProcessor


def _normalize_vispruner_prune_score_mode(mode: str) -> str:
    m = str(mode).lower()
    if m == "vtt_matrix":
        return "vtt"
    return m


def _clip_self_attn_head_dim(self_attn: nn.Module) -> int:
    if hasattr(self_attn, "head_dim"):
        return int(self_attn.head_dim)
    embed = getattr(self_attn, "embed_dim", None)
    nh = getattr(self_attn, "num_heads", None) or getattr(self_attn, "num_attention_heads", None)
    if embed is not None and nh is not None:
        return int(embed) // int(nh)
    raise ValueError("无法从 CLIP self_attn 推断 head_dim")


def clip_euti_vtt_from_qk(
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    self_attn: nn.Module,
    hidden_states_out: torch.Tensor,
    hidden_states_in: torch.Tensor,
) -> torch.Tensor:
    """
    与 modeling_llama_pdrop.euti_attn_norm_scores_one_layer 一致：前缀上 Q 的均值、softmax(Q̄K^T/√d)、
    取 patch 列、×(该层每个 patch 上 ‖h_out‖_2 / ‖h_in‖_2)。h_in 为该层块输入、
    h_out 为块输出，对应 output_hidden_states 中相邻下标。q_flat/k_flat 为 CLIPAttention 的 q_proj/k_proj (B, L, D)。
    """
    B, L, _ = q_flat.shape
    head_dim = _clip_self_attn_head_dim(self_attn)
    nh = getattr(self_attn, "num_heads", None) or getattr(self_attn, "num_attention_heads", None)
    nh = int(nh)
    q = q_flat.view(B, L, nh, head_dim).transpose(1, 2)
    k = k_flat.view(B, L, nh, head_dim).transpose(1, 2)
    device = q.device
    # ViT：0=CLS，1..L-1=patch；与 VisPruner patch 特征对齐
    is_image = torch.zeros(L, dtype=torch.bool, device=device)
    if L > 1:
        is_image[1:] = True
    last_true_idx = L - 1
    is_not_text = torch.zeros(L, dtype=torch.bool, device=device)
    is_not_text[: last_true_idx + 1] = True
    current_q = q[:, :, is_not_text, :]
    current_k = k[:, :, is_not_text, :]
    global_q = torch.mean(current_q, dim=2, keepdim=True)
    scores = torch.matmul(
        global_q.to(torch.float32) / math.sqrt(head_dim),
        current_k.to(torch.float32).transpose(-2, -1),
    )
    scores = F.softmax(scores, dim=-1, dtype=torch.float32).mean(dim=1)
    scores = torch.nan_to_num(scores, nan=0.0)
    scores = scores.squeeze(-2)
    is_image_sub = is_image[is_not_text]
    scores = scores[:, is_image_sub]
    eps = 1e-8
    h_out = hidden_states_out[:, 1:].to(torch.float32)
    h_in = hidden_states_in[:, 1:].to(torch.float32)
    patch_norm = torch.norm(h_out, dim=-1, p=2) / (torch.norm(h_in, dim=-1, p=2) + eps)
    scores = scores * patch_norm
    return scores.to(hidden_states_out.dtype)


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        # vtt：hook 取 Q/K 按 EUTI；失败时回退为对 softmax 注意力矩阵均值×‖h‖；cls：CLS→patch
        self.mm_vispruner_prune_score = getattr(args, "mm_vispruner_prune_score", "vtt")
        # 热力图专用 ViT encoder 层下标（0 起，支持负数）；None 时与 mm_vision_select_layer 相同
        self.mm_vision_heatmap_layer = getattr(args, "mm_vision_heatmap_layer", None)

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def _clip_encoder_layers(self):
        """
        兼容 transformers 版本：新版 CLIPVisionModel 为 vision_model.encoder.layers，
        旧版为顶层 encoder.layers。
        """
        vm = self.vision_tower
        if hasattr(vm, "vision_model") and hasattr(vm.vision_model, "encoder"):
            return vm.vision_model.encoder.layers
        if hasattr(vm, "encoder"):
            return vm.encoder.layers
        raise AttributeError(
            "CLIPVisionModel 上找不到 encoder.layers（已尝试 vision_model.encoder 与 encoder）"
        )

    def _resolve_vit_layer_index(self, sl) -> int:
        """将 ViT encoder 层下标（可为负，与 select_layer 约定一致）解析为 0..n-1。"""
        n = len(self._clip_encoder_layers())
        s = int(sl)
        if s < 0:
            s = n + s
        if s < 0 or s >= n:
            raise ValueError(f"ViT 层下标越界: 请求={sl!r} -> {s!r} (共 {n} 层 encoder)")
        return s

    def _resolve_encoder_layer_idx(self) -> int:
        return self._resolve_vit_layer_index(self.select_layer)

    def _resolve_heatmap_layer_index(self) -> int:
        """热力图所画层：运行时可 _vit_heatmap_layer 覆盖；否则 mm_vision_heatmap_layer；再否则 select_layer。"""
        sl = getattr(self, "_vit_heatmap_layer", None)
        if sl is not None:
            return self._resolve_vit_layer_index(sl)
        sl = getattr(self, "mm_vision_heatmap_layer", None)
        if sl is not None:
            return self._resolve_vit_layer_index(sl)
        return self._resolve_encoder_layer_idx()

    def _try_forward_vtt_hook_euti(self, images: torch.Tensor):
        """
        在选定 encoder 层对 q_proj/k_proj 注册 forward hook，与同一次 vision 前向共用；
        用 Q、K 按 EUTI 计算 VTT。返回 (patch_features, vtt) 或 None（失败时由上层回退）。
        """
        try:
            layer_idx = self._resolve_encoder_layer_idx()
            layers = self._clip_encoder_layers()
            self_attn = layers[layer_idx].self_attn
        except (ValueError, AttributeError):
            return None
        vm = self.vision_tower
        storage = {}

        def _hook(key):
            def _fn(_m, _inp, out):
                storage[key] = out.detach()

            return _fn

        h1 = self_attn.q_proj.register_forward_hook(_hook("q"))
        h2 = self_attn.k_proj.register_forward_hook(_hook("k"))
        try:
            out = vm(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
                output_attentions=False,
            )
        finally:
            h1.remove()
            h2.remove()
        if storage.get("q") is None or storage.get("k") is None:
            return None
        hs_in = out.hidden_states[layer_idx]
        hs_out = out.hidden_states[layer_idx + 1]
        vtt = clip_euti_vtt_from_qk(storage["q"], storage["k"], self_attn, hs_out, hs_in)
        patch_features = out.hidden_states[self.select_layer][:, 1:]
        return patch_features, vtt

    def _forward_vit_prune_layer_heatmaps(self, images: torch.Tensor):
        """
        一次前向：在 _resolve_heatmap_layer_index 指定层上 hook Q/K + output_attentions，
        画该层 CLS→patch 与 EUTI-VTT 热力图；再按 feature_select（仍用 select_layer）返回剪枝用特征。
        """
        try:
            layer_idx = self._resolve_heatmap_layer_index()
            layers = self._clip_encoder_layers()
            self_attn = layers[layer_idx].self_attn
        except (ValueError, AttributeError):
            return None
        cls_dir = getattr(self, "_vit_heatmap_cls_dir", None)
        vtt_dir = getattr(self, "_vit_heatmap_vtt_dir", None)
        if not cls_dir and not vtt_dir:
            return None
        vm = self.vision_tower
        storage = {}

        def _hook(key):
            def _fn(_m, _inp, out):
                storage[key] = out.detach()

            return _fn

        h1 = self_attn.q_proj.register_forward_hook(_hook("q"))
        h2 = self_attn.k_proj.register_forward_hook(_hook("k"))
        try:
            out = vm(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
                output_attentions=True,
            )
        finally:
            h1.remove()
            h2.remove()
        if storage.get("q") is None or storage.get("k") is None:
            return None
        hi = layer_idx
        attn_layer = out.attentions[hi]
        cls_b = attn_layer[:, :, 0, 1:].float().mean(dim=1)
        hs_in = out.hidden_states[hi]
        hs_out = out.hidden_states[hi + 1]
        vtt_b = clip_euti_vtt_from_qk(storage["q"], storage["k"], self_attn, hs_out, hs_in)
        from llava.eval.vit_vispruner_heatmaps import save_vit_prune_heatmaps_batch

        save_vit_prune_heatmaps_batch(
            cls_b,
            vtt_b,
            self.num_patches_per_side,
            cls_dir,
            vtt_dir,
            getattr(self, "_vit_heatmap_qid", None),
        )
        image_features, image_attentions = self.feature_select(out, output_attentions=True)
        return image_features.to(images.dtype), image_attentions

    # [VisPruner] Select image features and attentions
    def feature_select(self, image_forward_outs, output_attentions=False):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if output_attentions:
            image_attentions = image_forward_outs.attentions[self.select_layer]
        if self.select_feature == 'patch':
            if output_attentions:
                mode = _normalize_vispruner_prune_score_mode(
                    getattr(self, "mm_vispruner_prune_score", "vtt")
                )
                if mode == "cls":
                    image_attentions = image_attentions[:, :, 0, 1:]
                elif mode == "vtt":
                    # hook 失败时回退：对 softmax 权重矩阵在 query/head 上均值 × (‖h_out‖/‖h_in‖)
                    attn = image_attentions
                    attn_pp = attn[:, :, 1:, 1:].float()
                    scores = attn_pp.mean(dim=2).mean(dim=1)
                    sl = self._resolve_vit_layer_index(self.select_layer)
                    hs = image_forward_outs.hidden_states
                    h_in = hs[sl][:, 1:].float()
                    h_out = hs[sl + 1][:, 1:].float()
                    eps = 1e-8
                    norm = torch.norm(h_out, dim=-1, p=2) / (torch.norm(h_in, dim=-1, p=2) + eps)
                    image_attentions = scores * norm
                else:
                    image_attentions = image_attentions[:, :, 0, 1:]
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
            if output_attentions:
                image_attentions = image_attentions
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        if output_attentions:
            return image_features, image_attentions
        return image_features

    # [VisPruner] Get image features and attentions
    @torch.no_grad()
    def forward(self, images, output_attentions=False):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            mode = _normalize_vispruner_prune_score_mode(
                getattr(self, "mm_vispruner_prune_score", "vtt")
            )
            need_hm = bool(
                getattr(self, "_vit_heatmap_cls_dir", None) or getattr(self, "_vit_heatmap_vtt_dir", None)
            )
            if need_hm and output_attentions and self.select_feature == "patch":
                packed = self._forward_vit_prune_layer_heatmaps(images)
                if packed is not None:
                    return packed
            if (
                (not need_hm)
                and output_attentions
                and self.select_feature == "patch"
                and mode == "vtt"
            ):
                packed = self._try_forward_vtt_hook_euti(images)
                if packed is not None:
                    image_features, image_attentions = packed
                    return image_features.to(images.dtype), image_attentions
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype),
                                                   output_hidden_states=True, output_attentions=output_attentions)
            image_features = self.feature_select(image_forward_outs, output_attentions=output_attentions)
            if output_attentions:
                image_features = image_features[0].to(images.dtype), image_features[1]
            else:
                image_features = image_features.to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
