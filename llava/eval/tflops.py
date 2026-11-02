# ReGainDrop release preparation (2026-11-02): distribution notice added.
# Implementation retained from the author-supplied development snapshot; see NOTICE.
from dataclasses import dataclass

import torch.nn as nn


FLOPS_PER_TFLOP = 1e12


def _as_int(value, default=0):
    if value is None:
        return default
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def _get_config_int(config, name, default=None):
    value = getattr(config, name, default)
    if value is None:
        return default
    return int(value)


def _linear_flops(tokens, in_features, out_features):
    return 2.0 * float(tokens) * float(in_features) * float(out_features)


def _estimate_llama_generation_flops(config, prompt_tokens, generated_tokens, num_beams=1):
    layers = _get_config_int(config, "num_hidden_layers")
    hidden = _get_config_int(config, "hidden_size")
    intermediate = _get_config_int(config, "intermediate_size")
    vocab = _get_config_int(config, "vocab_size")
    if not all([layers, hidden, intermediate, vocab]):
        return 0.0

    prompt_tokens = max(_as_int(prompt_tokens), 0)
    generated_tokens = max(_as_int(generated_tokens), 0)
    num_beams = max(_as_int(num_beams, 1), 1)

    # LLaMA block: qkv/o projections, SwiGLU MLP, and QK/AV attention matmuls.
    prefill_per_layer = (
        8.0 * prompt_tokens * hidden * hidden
        + 6.0 * prompt_tokens * hidden * intermediate
        + 4.0 * prompt_tokens * prompt_tokens * hidden
    )
    prefill_lm_head = _linear_flops(prompt_tokens, hidden, vocab)

    # The first generated token is produced from prefill logits; cached decode
    # starts from the second generated token.
    decode_steps = max(generated_tokens - 1, 0)
    decode_context_sum = (
        decode_steps * prompt_tokens
        + decode_steps * (decode_steps + 1) / 2.0
    )
    decode_per_layer = (
        decode_steps * (8.0 * hidden * hidden + 6.0 * hidden * intermediate)
        + 4.0 * decode_context_sum * hidden
    )
    decode_lm_head = _linear_flops(decode_steps, hidden, vocab)

    return num_beams * (
        layers * (prefill_per_layer + decode_per_layer)
        + prefill_lm_head
        + decode_lm_head
    )


def _estimate_clip_vit_flops_for_size(config, image_size):
    layers = _get_config_int(config, "num_hidden_layers")
    hidden = _get_config_int(config, "hidden_size")
    intermediate = _get_config_int(config, "intermediate_size")
    patch_size = _get_config_int(config, "patch_size")
    channels = _get_config_int(config, "num_channels", 3)
    image_size = _as_int(image_size)
    if not all([layers, hidden, intermediate, patch_size, channels, image_size]):
        return 0.0

    patches_per_side = image_size // patch_size
    patch_tokens = patches_per_side * patches_per_side
    seq_tokens = patch_tokens + 1
    patch_embed = _linear_flops(
        patch_tokens,
        patch_size * patch_size * channels,
        hidden,
    )
    encoder_per_layer = (
        8.0 * seq_tokens * hidden * hidden
        + 6.0 * seq_tokens * hidden * intermediate
        + 4.0 * seq_tokens * seq_tokens * hidden
    )
    return patch_embed + layers * encoder_per_layer


def _estimate_clip_vit_flops(vision_tower, num_images):
    if vision_tower is None or num_images <= 0:
        return 0.0
    config = getattr(vision_tower, "config", None)
    if config is None:
        return 0.0

    scales = getattr(vision_tower, "s2_scales", None)
    if scales:
        per_image = sum(_estimate_clip_vit_flops_for_size(config, s) for s in scales)
    else:
        per_image = _estimate_clip_vit_flops_for_size(
            config,
            _get_config_int(config, "image_size"),
        )
    return float(num_images) * per_image


def _raw_projector_tokens(vision_tower, num_images):
    if vision_tower is None or num_images <= 0:
        return 0
    if hasattr(vision_tower, "s2_scales"):
        config = getattr(vision_tower, "config", None)
        patch_size = _get_config_int(config, "patch_size") if config is not None else None
        if patch_size:
            return int(num_images) * sum((int(s) // patch_size) ** 2 for s in vision_tower.s2_scales)
    raw_tokens = getattr(vision_tower, "num_patches", None)
    if raw_tokens is None:
        return 0
    if getattr(vision_tower, "select_feature", "patch") == "cls_patch":
        raw_tokens += 1
    return int(num_images) * int(raw_tokens)


def _estimate_projector_flops(mm_projector, tokens):
    if mm_projector is None or tokens <= 0:
        return 0.0
    total = 0.0
    for module in mm_projector.modules():
        if isinstance(module, nn.Linear):
            total += _linear_flops(tokens, module.in_features, module.out_features)
    return total


@dataclass
class LlavaTFlopsEstimate:
    prompt_tokens: int
    generated_tokens: int
    visual_tokens: int
    num_images: int
    language_flops: float
    vision_flops: float
    projector_flops: float

    @property
    def total_flops(self):
        return self.language_flops + self.vision_flops + self.projector_flops

    @property
    def total_tflops(self):
        return self.total_flops / FLOPS_PER_TFLOP


class TFlopsAccumulator:
    def __init__(self):
        self.samples = 0
        self.prompt_tokens = 0
        self.generated_tokens = 0
        self.visual_tokens = 0
        self.language_flops = 0.0
        self.vision_flops = 0.0
        self.projector_flops = 0.0
        self.generate_seconds = 0.0

    def update(self, estimate, generate_seconds):
        self.samples += 1
        self.prompt_tokens += estimate.prompt_tokens
        self.generated_tokens += estimate.generated_tokens
        self.visual_tokens += estimate.visual_tokens
        self.language_flops += estimate.language_flops
        self.vision_flops += estimate.vision_flops
        self.projector_flops += estimate.projector_flops
        self.generate_seconds += float(generate_seconds or 0.0)

    @property
    def total_flops(self):
        return self.language_flops + self.vision_flops + self.projector_flops

    def _avg(self, value):
        return value / self.samples if self.samples else 0.0

    def format_summary(self, e2e_seconds=None):
        avg_total_tflops = self._avg(self.total_flops) / FLOPS_PER_TFLOP
        avg_lang_tflops = self._avg(self.language_flops) / FLOPS_PER_TFLOP
        avg_vision_tflops = self._avg(self.vision_flops) / FLOPS_PER_TFLOP
        avg_projector_tflops = self._avg(self.projector_flops) / FLOPS_PER_TFLOP
        gen_tflops_s = (
            self.total_flops / self.generate_seconds / FLOPS_PER_TFLOP
            if self.generate_seconds > 0
            else 0.0
        )
        e2e_tflops_s = (
            self.total_flops / e2e_seconds / FLOPS_PER_TFLOP
            if e2e_seconds and e2e_seconds > 0
            else 0.0
        )
        return (
            "  compute_estimate (LLaVA1.5 theoretical): "
            f"avg_prompt_tokens={self._avg(self.prompt_tokens):.1f}, "
            f"avg_generated_tokens={self._avg(self.generated_tokens):.1f}, "
            f"avg_visual_tokens={self._avg(self.visual_tokens):.1f}\n"
            f"    avg_flops={avg_total_tflops:.4f} TFLOPs/sample "
            f"(llm={avg_lang_tflops:.4f}, vision={avg_vision_tflops:.4f}, "
            f"projector={avg_projector_tflops:.4f})\n"
            f"    achieved={gen_tflops_s:.3f} TFLOPs/s (generate_only), "
            f"{e2e_tflops_s:.3f} TFLOPs/s (e2e_after_load)"
        )


def estimate_llava15_inference_tflops(
    model,
    input_ids,
    output_ids,
    visual_token_num,
    image_token_index,
    num_beams=1,
):
    image_slots = int((input_ids == image_token_index).sum().item())
    visual_tokens = _as_int(visual_token_num) * image_slots
    prompt_tokens = int(input_ids.shape[-1]) - image_slots + visual_tokens
    generated_tokens = int(output_ids.shape[-1])

    language_flops = _estimate_llama_generation_flops(
        model.config,
        prompt_tokens,
        generated_tokens,
        num_beams=num_beams,
    )
    vision_tower = model.get_vision_tower() if hasattr(model, "get_vision_tower") else None
    vision_flops = _estimate_clip_vit_flops(vision_tower, image_slots)
    projector = getattr(model.get_model(), "mm_projector", None) if hasattr(model, "get_model") else None
    projector_flops = _estimate_projector_flops(
        projector,
        _raw_projector_tokens(vision_tower, image_slots),
    )

    return LlavaTFlopsEstimate(
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        visual_tokens=visual_tokens,
        num_images=image_slots,
        language_flops=language_flops,
        vision_flops=vision_flops,
        projector_flops=projector_flops,
    )
