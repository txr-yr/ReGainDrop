# ReGainDrop release preparation (2026-11-02): distribution notice added.
# Implementation retained from the author-supplied development snapshot; see NOTICE.
# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import math
import warnings
from typing import List, Optional, Tuple, Union
import copy

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.utils.import_utils import is_torch_fx_available
from transformers.models.llama.configuration_llama import LlamaConfig

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    warnings.warn(
        "Calling `transformers.models.llama.modeling_llama._prepare_4d_attention_mask` is deprecated and will be removed in v4.37. Use `transformers.modeling_attn_mask_utils._prepare_4d_attention_mask"
    )
    return _prepare_4d_attention_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    warnings.warn(
        "Calling `transformers.models.llama.modeling_llama._make_causal_mask` is deprecated and will be removed in v4.37. Use `transformers.models.llama.modeling_llama.AttentionMaskConverter._make_causal_mask"
    )
    return AttentionMaskConverter._make_causal_mask(
        input_ids_shape=input_ids_shape, dtype=dtype, device=device, past_key_values_length=past_key_values_length
    )


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # 防止 position_ids 越界（例如 rank_drop 后序列缩短但 position_ids 未同步）
    # cos 用于索引的维度：2D 为 dim0，3D 为 dim1
    cos_seq_dim = 1 if cos.dim() == 3 else 0
    cos_len = cos.shape[cos_seq_dim]
    if position_ids is not None:
        if isinstance(position_ids, (list, tuple)):
            position_ids = torch.stack([p for p in position_ids], dim=0).to(cos.device)
        if isinstance(position_ids, torch.Tensor):
            if position_ids.shape[-1] > cos_len:
                position_ids = position_ids[..., :cos_len].contiguous()
            position_ids = position_ids.clamp(0, cos_len - 1)
    # 索引 cos/sin 的序列维（与 position_ids 对应），输出形状与 q/k 可广播
    if cos.dim() == 3:
        cos = cos[:, position_ids, :].squeeze(0).unsqueeze(unsqueeze_dim)
        sin = sin[:, position_ids, :].squeeze(0).unsqueeze(unsqueeze_dim)
    else:
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def euti_attn_norm_scores_one_layer(
    self_attn: nn.Module,
    hidden_states_in: torch.Tensor,
    hidden_states_normed: torch.Tensor,
    position_ids: torch.Tensor,
    image_index: int,
    cur_image_count: int,
    use_rope: bool = True,
    hidden_states_for_norm: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    与 layer_vtt_heatmap.py 一致：is_not_text 前缀、mean_q、可选 RoPE、×‖h_norm_src‖_2；
    QK^T 除以 √head_dim。返回 (q_len,) 向量，非视觉位为 nan。
    """
    if getattr(self_attn.config, "pretraining_tp", 1) > 1:
        raise NotImplementedError("euti_attn_norm_scores_one_layer 不支持 pretraining_tp > 1")

    bsz, q_len, _ = hidden_states_normed.size()
    if bsz != 1:
        raise ValueError("euti_attn_norm_scores_one_layer 当前仅支持 batch_size=1")

    end_img = image_index + cur_image_count
    device = hidden_states_normed.device
    is_image = torch.zeros(q_len, dtype=torch.bool, device=device)
    if image_index < end_img and image_index >= 0:
        is_image[image_index:end_img] = True

    is_image_idx = torch.where(is_image)[0]
    if is_image_idx.numel() == 0:
        raise ValueError("is_image 为空")

    last_true_idx = int(is_image_idx.max().item())
    is_not_text = torch.zeros(q_len, dtype=torch.bool, device=device)
    is_not_text[: last_true_idx + 1] = True

    query_states = self_attn.q_proj(hidden_states_normed)
    key_states = self_attn.k_proj(hidden_states_normed)

    query_states = query_states.view(bsz, q_len, self_attn.num_heads, self_attn.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self_attn.num_key_value_heads, self_attn.head_dim).transpose(1, 2)

    if use_rope:
        value_states = self_attn.v_proj(hidden_states_normed)
        value_states = value_states.view(bsz, q_len, self_attn.num_key_value_heads, self_attn.head_dim).transpose(
            1, 2
        )
        kv_seq_len = key_states.shape[-2]
        cos, sin = self_attn.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    current_q_image = query_states[:, :, is_not_text, :]
    current_k_image = key_states[:, :, is_not_text, :]
    n_kv_groups = getattr(
        self_attn, "num_key_value_groups", self_attn.num_heads // self_attn.num_key_value_heads
    )
    current_k_image = repeat_kv(current_k_image, n_kv_groups)

    global_q = torch.mean(current_q_image, dim=2, keepdim=True)
    head_dim = self_attn.head_dim
    scores = torch.matmul(
        global_q.to(torch.float32) / math.sqrt(head_dim),
        current_k_image.to(torch.float32).transpose(-2, -1),
    )
    scores = torch.softmax(scores, dim=-1, dtype=torch.float32).mean(dim=1)
    scores = torch.nan_to_num(scores, nan=0.0)
    scores = scores.squeeze(-2)

    is_image_sub = is_image[is_not_text]
    scores = scores[:, is_image_sub]

    norm_src = hidden_states_in if hidden_states_for_norm is None else hidden_states_for_norm
    norm = torch.norm(norm_src[:, is_image].to(torch.float32), dim=-1, p=2).squeeze(0)
    scores = scores.to(torch.float32).squeeze(0) * norm

    out = torch.full((q_len,), float("nan"), dtype=torch.float32, device=device)
    out[is_image] = scores.to(out.dtype)
    return out


def compute_layer_vtt_metric_vector(
    self_attn: nn.Module,
    hidden_states_in: torch.Tensor,
    hidden_states_normed: torch.Tensor,
    position_ids: torch.Tensor,
    image_index: int,
    cur_image_count: int,
    attention_mask_row: torch.Tensor,
    hidden_states_out: Optional[torch.Tensor] = None,
    *,
    use_rope: bool = True,
) -> torch.Tensor:
    """
    layer_vtt_heatmap.py 中每层指标的完整计算（EUTI 式）。
    注意：若提供 hidden_states_out，则范数项改为 ‖h_out‖_2；否则回退为 ‖h_in‖_2。
    """
    _ = attention_mask_row  # 保留参数以兼容旧调用签名
    S = hidden_states_in.shape[1]
    end = min(image_index + cur_image_count, S)
    n_vis = end - image_index
    if n_vis <= 0:
        return torch.full((S,), float("nan"), device=hidden_states_in.device, dtype=torch.float32)

    return euti_attn_norm_scores_one_layer(
        self_attn,
        hidden_states_in,
        hidden_states_normed,
        position_ids,
        image_index,
        n_vis,
        use_rope=use_rope,
        hidden_states_for_norm=hidden_states_out,
    )


def compute_last_text_to_visual_attn_scores_for_rank(
    self_attn: nn.Module,
    hidden_states_normed: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask_2d: torch.Tensor,
    image_index: int,
    cur_image_count: int,
    *,
    sliding_window: Optional[int] = None,
) -> torch.Tensor:
    """
    hired_pdrop_last：用「当前序列中最后一个文本 token」在标准因果注意力下对全体 key 的分布，
    再取图像区间上的权重（多头平均）作为各视觉 token 的排序分数。
    与 layer_last_text_attn_heatmap._self_attn_row_one_query 一致（含 4d causal mask）。
    """
    if getattr(self_attn.config, "pretraining_tp", 1) > 1:
        raise NotImplementedError("compute_last_text_to_visual_attn_scores_for_rank 不支持 pretraining_tp > 1")

    bsz, q_len, _ = hidden_states_normed.size()
    if bsz != 1:
        raise ValueError("compute_last_text_to_visual_attn_scores_for_rank 当前仅支持 batch_size=1")

    device = hidden_states_normed.device
    end_img = image_index + cur_image_count
    attn_bool = attention_mask_2d.to(device).bool()
    if attn_bool.dim() == 1:
        attn_bool = attn_bool.unsqueeze(0)

    is_image = torch.zeros(q_len, dtype=torch.bool, device=device)
    if 0 <= image_index < end_img <= q_len:
        is_image[image_index:end_img] = True

    is_text = attn_bool[0] & (~is_image)
    text_idx = torch.where(is_text)[0]
    if text_idx.numel() == 0:
        return torch.full((q_len,), float("nan"), dtype=torch.float32, device=device)

    last_text_idx = int(text_idx.max().item())

    layer_attn_mask = _prepare_4d_causal_attention_mask(
        attn_bool,
        (bsz, q_len),
        hidden_states_normed,
        past_key_values_length=0,
        sliding_window=sliding_window,
    )

    dtype = hidden_states_normed.dtype
    query_states = self_attn.q_proj(hidden_states_normed)
    key_states = self_attn.k_proj(hidden_states_normed)
    value_states = self_attn.v_proj(hidden_states_normed)

    query_states = query_states.view(bsz, q_len, self_attn.num_heads, self_attn.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self_attn.num_key_value_heads, self_attn.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self_attn.num_key_value_heads, self_attn.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    cos, sin = self_attn.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    n_kv_groups = getattr(
        self_attn, "num_key_value_groups", self_attn.num_heads // self_attn.num_key_value_heads
    )
    key_states = repeat_kv(key_states, n_kv_groups)

    head_dim = self_attn.head_dim
    q_row = query_states[:, :, last_text_idx : last_text_idx + 1, :]
    attn_scores = torch.matmul(q_row, key_states.transpose(2, 3)) / math.sqrt(head_dim)
    row_mask = layer_attn_mask[:, :, last_text_idx : last_text_idx + 1, :]
    attn_scores = attn_scores + row_mask
    attn_probs = F.softmax(attn_scores.float(), dim=-1).to(dtype)
    attn_probs = attn_probs.squeeze(2)
    row_mean = attn_probs[0].float().mean(dim=0)

    out = torch.full((q_len,), float("nan"), dtype=torch.float32, device=device)
    if 0 <= image_index < end_img <= q_len:
        out[image_index:end_img] = row_mean[image_index:end_img].to(out.dtype)
    return out


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaFlashAttention2(LlamaAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # LlamaFlashAttention2 attention does not support output_attentions
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class LlamaSdpaAttention(LlamaAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    
    def pdrop_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache  # False

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  #True

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if past_key_values_length == 0 and getattr(self, "_save_pdrop_drop_visual", False):
            self._pdrop_drop_visual = None

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for layer_idx, decoder_layer in enumerate(self.layers):
            # 保证 position_ids 与当前序列长度一致，避免 rank_drop 后 cos[position_ids] 越界
            cur_seq_len = hidden_states.shape[1]
            if (
                position_ids is not None
                and isinstance(position_ids, torch.Tensor)
                and position_ids.dim() >= 2
                and position_ids.shape[1] > cur_seq_len
            ):
                position_ids = position_ids[:, :cur_seq_len].contiguous()

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)


            # rank & drop after specific layer
            # only drop in prefill stage when inference
            rank_layer = layer_idx+1
            if rank_layer in self.layer_list:
                if hidden_states.shape[1]!=1:  # prefill stage or training
                    stage = self.layer_list.index(rank_layer) # determine current stage
                    (
                        position_ids,
                        attention_mask,
                        hidden_states,
                        labels   # update labels and return
                    ) = self.pdrop_rank_drop(    
                        cur_num = stage,
                        rank_layer = rank_layer,
                        features = hidden_states,   
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        labels = labels
                    )
                    # 序列已缩短，用当前长度重新生成 mask，避免 cos[position_ids] 等索引越界
                    seq_length = hidden_states.shape[1]
                    if position_ids is not None and position_ids.numel() > 0:
                        position_ids = position_ids.clamp(0, seq_length - 1)
                    # process attention_mask again after undating 
                    if self._use_flash_attention_2:
                        # 2d mask is passed through the layers
                        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
                    elif self._use_sdpa and not output_attentions:
                        # output_attentions=True can not be supported when using SDPA, and we fall back on
                        # the manual implementation that requires a 4D causal mask in all cases.
                        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                            attention_mask,
                            (batch_size, seq_length),
                            hidden_states,
                            past_key_values_length,
                        )
                    else:
                        # 4d mask is passed through the layers
                        attention_mask = _prepare_4d_causal_attention_mask(
                            attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
                        )
                        
                else:
                    # update position_ids in decoding stage when inference
                    stage = self.layer_list.index(rank_layer)
                    use_hired_decode = (
                        getattr(self, 'hired_pdrop', False)
                        and getattr(self, 'hired_actual_image_tokens_history', None) is not None
                        and stage + 1 < len(self.hired_actual_image_tokens_history)
                    )
                    if use_hired_decode:
                        cur_visual_length = [self.hired_actual_image_tokens_history[stage][idx] for idx in range(len(self.image_tokens))]
                        next_visual_length = [self.hired_actual_image_tokens_history[stage + 1][idx] for idx in range(len(self.image_tokens))]
                    else:
                        cur_visual_length = [int(cur_image_token * self.image_token_ratio_list[stage]) for cur_image_token in self.image_tokens]
                        next_visual_length = [int(cur_image_token * self.image_token_ratio_list[stage + 1]) for cur_image_token in self.image_tokens]
                    new_position_ids = []
                    for idx, cur_position_ids in enumerate(position_ids):
                        cur_position_ids = cur_position_ids - (cur_visual_length[idx] - next_visual_length[idx])
                        new_position_ids.append(cur_position_ids)
                    position_ids = new_position_ids

        hidden_states = self.norm(hidden_states)   

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None),labels
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        ),labels

    # implementation of pdrop
    def pdrop_rank_drop(
        self, cur_num, rank_layer, features ,
        position_ids, attention_mask, labels
    ):

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask

        if position_ids is None:
            position_ids = torch.arange(0, features.shape[1], dtype=torch.long, device=features.device).unsqueeze(0)
        
        if getattr(self.config, 'tokenizer_padding_side', 'right') == "right":
            
            batch_size = features.shape[0]
            image_tokens = [int(cur_image_token * self.image_token_ratio_list[cur_num]) for cur_image_token in self.image_tokens]
            keep_length = [int(cur_image_token * self.image_token_ratio_list[cur_num + 1]) for cur_image_token in self.image_tokens]

            # Random drop: 在指定 rank_layer 对图像 token 随机保留指定比例。
            # 该逻辑不依赖 pdrop/hired_pdrop 的排序指标（VTT/EUIT），只做随机抽样。
            random_drop_enabled = bool(getattr(self, "random_drop_enabled", False))
            random_drop_layer = getattr(self, "random_drop_layer", None)
            random_keep_ratio = float(getattr(self, "random_keep_ratio", 0.5))
            random_drop_seed = getattr(self, "random_drop_seed", 0)
            if random_keep_ratio < 0.0:
                random_keep_ratio = 0.0
            if random_keep_ratio > 1.0:
                random_keep_ratio = 1.0
            apply_random_drop = (
                random_drop_enabled
                and random_drop_layer is not None
                and int(rank_layer) == int(random_drop_layer)
            )

            # 热力图记录：4 个槽位 = 第8层、第16层、第24层、最后一层(同第24层后)
            if getattr(self, 'hired_record_heatmap', False):
                hm = getattr(self, 'hired_heatmap_masks', None)
                if hm is None:
                    self.hired_heatmap_masks = [None, None, None, None]
                    hm = self.hired_heatmap_masks
                if hm[cur_num] is None:
                    hm[cur_num] = [None] * batch_size
                if cur_num == len(self.layer_list) - 1:
                    if len(hm) <= len(self.layer_list):
                        hm.append([None] * batch_size)
                    elif hm[len(self.layer_list)] is None:
                        hm[len(self.layer_list)] = [None] * batch_size

            features_list = []
            attention_mask_list = []
            labels_list = []

            if attention_mask is None:
                attention_mask = torch.ones((batch_size,features.shape[1]), dtype=torch.bool, device=features.device)
            else:
                attention_mask = attention_mask.bool()
            if labels is None:
                labels = torch.full((batch_size,features.shape[1]), IGNORE_INDEX, device=features.device)            

            # 该层完整前向一次，得到 EUIT 所需的 h_in/h_out（与主 forward 同 mask/pos）
            features_btt = features.clone().detach()
            sliding_window = getattr(self.config, "sliding_window", None)
            layer_attn_mask = attention_mask
            if attention_mask is not None and attention_mask.dim() == 2:
                layer_attn_mask = _prepare_4d_causal_attention_mask(
                    attention_mask,
                    (batch_size, features.shape[1]),
                    features_btt,
                    past_key_values_length=0,
                    sliding_window=sliding_window,
                )
            with torch.no_grad():
                layer_out_btt = self.layers[rank_layer](
                    features_btt,
                    attention_mask=layer_attn_mask,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                )[0]

            attention_mask_batched = attention_mask

            # take valid features (and valid sub_image_ids before overwriting attention_mask)
            sub_image_ids_list = None
            patch_index_list = None
            if getattr(self, 'sub_image_ids', None) is not None:
                sub_image_ids_list = [self.sub_image_ids[i][attention_mask[i]] for i in range(batch_size)]
            if getattr(self, 'sub_image_patch_index', None) is not None:
                patch_index_list = [self.sub_image_patch_index[i][attention_mask[i]] for i in range(batch_size)]
            new_sub_image_ids_list = [] if sub_image_ids_list is not None else None
            new_patch_index_list = [] if patch_index_list is not None else None

            features = [cur_features[cur_attention_mask] for cur_features, cur_attention_mask in zip(features_btt, attention_mask)]
            layer_out_per_sample = [
                cur_o[cur_m] for cur_o, cur_m in zip(layer_out_btt, attention_mask)
            ]
            labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
            attention_mask = [cur_attention_mask[cur_attention_mask] for cur_attention_mask, cur_attention_mask in zip(attention_mask, attention_mask)]

            # HiRED: 使用实际 image token 数，而非 ratio 估算（hired 下实际保留数由 budget 决定）
            use_hired_global = (
                getattr(self, 'hired_pdrop', False)
                and getattr(self, 'hired_computed_budgets', None) is not None
                and sub_image_ids_list is not None
            )
            if use_hired_global and cur_num == 0:
                self.hired_actual_image_tokens = list(self.image_tokens)
                self.hired_actual_image_tokens_history = [list(self.image_tokens)]
                # 记录每个子图的原始 token 数，用于「每层丢弃 (original-budget)/3」策略
                self.hired_originals_per_subgraph = [dict() for _ in range(batch_size)]

            # rank & drop
            for i in range(batch_size):
                image_index= self.image_token_posi[i]
                if image_index == -1:
                    if use_hired_global and i < len(getattr(self, 'hired_actual_image_tokens', [])):
                        self.hired_actual_image_tokens[i] = 0
                    cur_input_embeds = features[i]
                    features_list.append(cur_input_embeds)
                    attention_mask_list.append(attention_mask[i])
                    labels_list.append(labels[i])
                    if new_sub_image_ids_list is not None:
                        new_sub_image_ids_list.append(sub_image_ids_list[i])
                    if new_patch_index_list is not None:
                        new_patch_index_list.append(patch_index_list[i])
                    continue
                
                # 训练时若无有效监督位置则跳过（与旧逻辑一致）
                if self.training:
                    answer_index = torch.where(labels[i] != -100)[0].tolist()
                    index_before_answer = []
                    for index in answer_index:
                        if labels[i][index - 1] == -100:
                            index_before_answer.append(index - 1)
                    if index_before_answer == []:
                        print("========index_before_answer is []")
                        cur_input_embeds = features[i]
                        features_list.append(cur_input_embeds)
                        attention_mask_list.append(attention_mask[i])
                        labels_list.append(labels[i])
                        continue

                # HiRED: 用实际 image token 数做切片；非 hired 用 ratio 估算
                cur_image_count = (
                    self.hired_actual_image_tokens[i]
                    if use_hired_global and i < len(getattr(self, 'hired_actual_image_tokens', []))
                    else image_tokens[i]
                )

                if (
                    getattr(self, "_save_pdrop_drop_visual", False)
                    and not self.training
                    and cur_num == 0
                ):
                    if getattr(self, "_pdrop_drop_visual", None) is None:
                        self._pdrop_drop_visual = [None] * batch_size
                    cnt0 = int(cur_image_count)
                    if patch_index_list is not None:
                        ip = patch_index_list[i][image_index : image_index + cnt0].detach().cpu()
                        ir = (
                            sub_image_ids_list[i][image_index : image_index + cnt0].detach().cpu()
                            if sub_image_ids_list is not None
                            else torch.zeros(cnt0, dtype=torch.long)
                        )
                    else:
                        ip = torch.arange(cnt0, dtype=torch.long)
                        ir = torch.zeros(cnt0, dtype=torch.long)
                    self._pdrop_drop_visual[i] = {"init_patch": ip, "init_real": ir}

                m_vis = attention_mask_batched[i, image_index : image_index + cur_image_count].float()
                m_b = m_vis.bool()

                vtt_scores_i = None
                if not apply_random_drop:
                    self_attn = self.layers[rank_layer].self_attn
                    normed = self.layers[rank_layer].input_layernorm(features_btt[i : i + 1])
                    # 默认：VTT（EUTI：RoPE、×‖h_out‖）。
                    # hired_pdrop_last：最后一枚文本 token 对图像的因果注意力；无效时回退 VTT。
                    # 随机丢弃层不计算排序分数。
                    if getattr(self, "hired_pdrop_last", False):
                        vtt_scores_i = compute_last_text_to_visual_attn_scores_for_rank(
                            self_attn,
                            normed,
                            position_ids[i : i + 1],
                            attention_mask_batched[i : i + 1],
                            int(image_index),
                            int(cur_image_count),
                            sliding_window=sliding_window,
                        )
                        if vtt_scores_i is not None:
                            vis_slice = vtt_scores_i[image_index : image_index + cur_image_count]
                            m_v = m_vis.to(vis_slice.device)
                            finite = torch.isfinite(vis_slice) & m_v.bool()
                            vis_sum = float(vis_slice[finite].sum().item()) if finite.any() else 0.0
                            if (not finite.any()) or (vis_sum < 1e-12):
                                vtt_scores_i = None
                    if vtt_scores_i is None:
                        vtt_scores_i = compute_layer_vtt_metric_vector(
                            self_attn,
                            features_btt[i : i + 1],
                            normed,
                            position_ids[i : i + 1],
                            int(image_index),
                            int(cur_image_count),
                            attention_mask_batched[i],
                            hidden_states_out=layer_out_per_sample[i].unsqueeze(0),
                            use_rope=False,
                        )

                use_hired = (
                    getattr(self, "hired_pdrop", False)
                    and getattr(self, "hired_computed_budgets", None) is not None
                    and i < len(self.hired_computed_budgets)
                    and sub_image_ids_list is not None
                )
                if use_hired:
                    cur_real = sub_image_ids_list[i][image_index : image_index + cur_image_count]
                    b = self.hired_computed_budgets[i]
                    full_img_budget = b["full_img_budget"]
                    sub_img_budgets = b.get("sub_img_budgets") or []
                    num_drop_layers = len(self.layer_list)
                    hired_originals = getattr(self, "hired_originals_per_subgraph", None)
                    if hired_originals is None:
                        hired_originals = [dict() for _ in range(batch_size)]
                    kept = []
                    if apply_random_drop:
                        # 随机保留：在全部图像 token 中随机抽取，而不按子图预算分组。
                        local_valid_rel = torch.where(m_b)[0]
                        n_local = int(local_valid_rel.numel())
                        if n_local == 0:
                            kept = []
                            top_rank_index = torch.empty((0,), dtype=torch.long, device=features[i].device)
                        else:
                            if random_keep_ratio <= 0.0:
                                k_local = 0
                            else:
                                k_local = max(1, int(n_local * random_keep_ratio))
                                k_local = min(k_local, n_local)
                            gen = torch.Generator(device=features[i].device)
                            # 让同一 seed 在不同样本/不同层可复现且有差异
                            gen.manual_seed(int(random_drop_seed) + i * 10007 + int(rank_layer) * 97)
                            perm = torch.randperm(n_local, generator=gen, device=features[i].device)
                            chosen_rel = local_valid_rel[perm[:k_local]]
                            top_rank_index = (chosen_rel + image_index).sort().values
                            kept = top_rank_index.tolist()
                    else:
                        for val in cur_real.unique().tolist():
                            val = int(val)
                            mask = cur_real == val
                            indices_local = mask.nonzero(as_tuple=True)[0]
                            if val == 0:
                                budget = full_img_budget
                            else:
                                budget = int(sub_img_budgets[val - 1]) if (sub_img_budgets and val - 1 < len(sub_img_budgets)) else 0
                            if cur_num == 0:
                                original = len(indices_local)
                                if i < len(hired_originals):
                                    hired_originals[i][val] = original
                            else:
                                original = (
                                    hired_originals[i].get(val, len(indices_local))
                                    if i < len(hired_originals)
                                    else len(indices_local)
                                )
                            total_drop = original - budget
                            if total_drop <= 0:
                                keep_at_this_layer = len(indices_local)
                            else:
                                drop_per_layer_base = total_drop // num_drop_layers
                                remainder = total_drop % num_drop_layers
                                drops = [drop_per_layer_base] * num_drop_layers
                                for r in range(remainder):
                                    drops[r] += 1
                                keep_at_this_layer = original - sum(drops[: cur_num + 1])
                            k = min(max(0, keep_at_this_layer), len(indices_local))
                            if k == 0:
                                continue
                            if vtt_scores_i is not None:
                                vtt_at = vtt_scores_i[image_index + indices_local]
                                _, top_idx = vtt_at.topk(k, largest=True)
                                kept.extend((image_index + indices_local[top_idx]).tolist())
                            else:
                                si = indices_local.sort().values[:k]
                                kept.extend((image_index + si).tolist())
                        top_rank_index = torch.tensor(kept, device=features[i].device, dtype=torch.long).sort().values
                else:
                    if apply_random_drop:
                        local_valid_rel = torch.where(m_b)[0]
                        n_local = int(local_valid_rel.numel())
                        if n_local == 0:
                            top_rank_index = torch.empty((0,), dtype=torch.long, device=features[i].device)
                            keep_length[i] = 0
                        else:
                            if random_keep_ratio <= 0.0:
                                k_local = 0
                            else:
                                k_local = max(1, int(n_local * random_keep_ratio))
                                k_local = min(k_local, n_local)
                            gen = torch.Generator(device=features[i].device)
                            gen.manual_seed(int(random_drop_seed) + i * 10007 + int(rank_layer) * 97)
                            perm = torch.randperm(n_local, generator=gen, device=features[i].device)
                            chosen_rel = local_valid_rel[perm[:k_local]]
                            top_rank_index = (chosen_rel + image_index).sort().values
                            keep_length[i] = k_local
                    else:
                        if vtt_scores_i is not None:
                            vtt_img = vtt_scores_i[image_index : image_index + cur_image_count]
                            vtt_img = (vtt_img * m_vis.to(vtt_img.dtype)).to(vtt_img.dtype)
                            top_rank_index = vtt_img.topk(keep_length[i]).indices + image_index
                        else:
                            valid_rel = m_b.nonzero(as_tuple=True)[0]
                            kk = min(int(keep_length[i]), int(valid_rel.numel()))
                            top_rank_index = valid_rel[:kk] + image_index
                        top_rank_index = top_rank_index.sort().values

                if (
                    getattr(self, "_save_pdrop_drop_visual", False)
                    and not self.training
                    and cur_num == len(self.layer_list) - 1
                ):
                    dv = getattr(self, "_pdrop_drop_visual", None)
                    if dv is not None and i < len(dv) and dv[i] is not None:
                        if patch_index_list is not None:
                            kp = patch_index_list[i][top_rank_index].detach().cpu()
                            kr = (
                                sub_image_ids_list[i][top_rank_index].detach().cpu()
                                if sub_image_ids_list is not None
                                else torch.zeros(top_rank_index.shape[0], dtype=torch.long)
                            )
                        else:
                            kp = (top_rank_index - int(image_index)).detach().cpu()
                            kr = torch.zeros(top_rank_index.shape[0], dtype=torch.long)
                        dv[i]["kept_patch"] = kp
                        dv[i]["kept_real"] = kr

                if use_hired:
                    self.hired_actual_image_tokens[i] = len(kept)
                elif use_hired_global and i < len(getattr(self, 'hired_actual_image_tokens', [])):
                    self.hired_actual_image_tokens[i] = keep_length[i]

                start_index = image_index + cur_image_count
                new_input_embeds = torch.cat([features[i][:image_index, :], features[i][top_rank_index, :], features[i][start_index:, :]], dim=0)
                new_labels = torch.cat([labels[i][:image_index], labels[i][top_rank_index], labels[i][start_index:]], dim=0)
                new_attention_mask = torch.cat([attention_mask[i][:image_index], attention_mask[i][top_rank_index], attention_mask[i][start_index:]], dim=0)

                features_list.append(new_input_embeds)
                attention_mask_list.append(new_attention_mask)
                labels_list.append(new_labels)
                if new_sub_image_ids_list is not None:
                    new_sub_image_ids_list.append(
                        torch.cat([sub_image_ids_list[i][:image_index], sub_image_ids_list[i][top_rank_index], sub_image_ids_list[i][start_index:]], dim=0)
                    )
                if new_patch_index_list is not None:
                    new_patch_index_list.append(
                        torch.cat([patch_index_list[i][:image_index], patch_index_list[i][top_rank_index], patch_index_list[i][start_index:]], dim=0)
                    )

                # 记录第 8/16/24 层及最后一层选择 token 的热力图 mask（用于推理脚本画图）
                if getattr(self, 'hired_record_heatmap', False) and use_hired and patch_index_list is not None:
                    hm = getattr(self, 'hired_heatmap_masks', None)
                    if hm is not None and cur_num < len(hm) and hm[cur_num] is not None:
                        reals_kept = sub_image_ids_list[i][top_rank_index]
                        patch_kept = patch_index_list[i][top_rank_index]
                        uniq = reals_kept.unique().tolist()
                        slot = {int(v): torch.zeros(576, dtype=torch.bool, device=reals_kept.device) for v in uniq}
                        for r, p in zip(reals_kept.tolist(), patch_kept.tolist()):
                            if 0 <= p < 576:
                                slot[r][p] = True
                        hm[cur_num][i] = {k: v.cpu() for k, v in slot.items()}
                        if cur_num == len(self.layer_list) - 1 and len(hm) > len(self.layer_list) and hm[len(self.layer_list)] is not None:
                            hm[len(self.layer_list)][i] = hm[cur_num][i]

            if use_hired_global:
                self.hired_actual_image_tokens_history.append(list(self.hired_actual_image_tokens))

            # Truncate sequences to max length as image embeddings can make the sequence longer
            tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', 2048)
            if tokenizer_model_max_length is not None:
                new_input_embeds = [x[:tokenizer_model_max_length] for x in features_list]
                new_attention_mask = [x[:tokenizer_model_max_length] for x in attention_mask_list]
                new_labels = [x[:tokenizer_model_max_length] for x in labels_list]
                if new_patch_index_list is not None:
                    new_patch_index_list = [x[:tokenizer_model_max_length] for x in new_patch_index_list]

            max_len = max(x.shape[0] for x in new_input_embeds)

            # padding the sequences to form batch
            embeds_padded=[]
            labels_paded=[]
            attention_mask_padded=[]
            position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
            sub_image_ids_padded = None
            patch_index_padded = None
            if new_sub_image_ids_list is not None:
                sub_image_ids_padded = torch.zeros((batch_size, max_len), dtype=torch.long, device=features[0].device)
            if new_patch_index_list is not None:
                patch_index_padded = torch.full((batch_size, max_len), -1, dtype=torch.long, device=features[0].device)
            for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
                cur_len_emb=cur_new_embed.shape[0]
                dif=max_len - cur_len_emb  # padding to longest seq
                
                cur_new_embed = torch.cat([cur_new_embed,torch.zeros((dif, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)],dim=0)
                cur_new_labels = torch.cat([cur_new_labels,torch.full((dif,),IGNORE_INDEX,dtype=cur_new_labels.dtype, device=cur_new_labels.device)],dim=0)
                cur_attention_mask = new_attention_mask[i]
                cur_attention_mask = torch.cat([cur_attention_mask,torch.full((dif,),False, dtype=cur_attention_mask.dtype, device=cur_attention_mask.device)],dim=0)
                
                embeds_padded.append(cur_new_embed)
                labels_paded.append(cur_new_labels)
                attention_mask_padded.append(cur_attention_mask)
                if sub_image_ids_padded is not None and i < len(new_sub_image_ids_list):
                    cur_len_real = new_sub_image_ids_list[i].shape[0]
                    sub_image_ids_padded[i, :cur_len_real] = new_sub_image_ids_list[i]
                if patch_index_padded is not None and i < len(new_patch_index_list):
                    cur_len_pi = new_patch_index_list[i].shape[0]
                    patch_index_padded[i, :cur_len_pi] = new_patch_index_list[i]

                cur_len = new_attention_mask[i].sum().item()
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            

            new_input_embeds = torch.stack(embeds_padded,dim=0)
            new_input_embeds = new_input_embeds.to(features[0].dtype)

            new_attention_mask = torch.stack(attention_mask_padded,dim=0)
            new_labels = torch.stack(labels_paded,dim=0)
            if sub_image_ids_padded is not None:
                self.sub_image_ids = sub_image_ids_padded
            if patch_index_padded is not None:
                self.sub_image_patch_index = patch_index_padded

            if _position_ids is None:
                position_ids = None
            if _labels is None:
                new_labels = None

            if _attention_mask is None:
                new_attention_mask = None
            else:
                new_attention_mask = new_attention_mask.to(dtype=_attention_mask.dtype)

            return position_ids, new_attention_mask, new_input_embeds, new_labels
        
        else:
            raise ValueError(f"Unexpected tokenizer_padding_side: {self.config.tokenizer_padding_side}")


class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
