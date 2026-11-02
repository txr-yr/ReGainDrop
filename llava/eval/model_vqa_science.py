# ReGainDrop release preparation (2026-11-02): distribution notice added.
# Implementation retained from the author-supplied development snapshot; see NOTICE.
import argparse
import json
import os
import time

import torch
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.eval.tflops import TFlopsAccumulator, estimate_llava15_inference_tflops
from llava.eval.vispruner_prune_vis import render_discarded_patches_on_original
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    t_load0 = time.perf_counter()
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name,
        visual_token_num=args.visual_token_num,
        important_ratio=args.important_ratio,
    )
    load_s = time.perf_counter() - t_load0
    model._vispruner_record_mask = bool(args.save_prune_vis_dir)
    model.config.mm_vispruner_spatial_sigma = float(args.mm_vispruner_spatial_sigma)

    # Data
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    gen_latencies = []
    tflops_meter = TFlopsAccumulator()
    e2e_t0 = time.perf_counter()

    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    data_bar = tqdm(questions)
    prune_vis_warned = False
    for i, line in enumerate(data_bar):
        idx = line["id"]
        question = line['conversations'][0]
        qs = question['value'].replace('<image>', '').strip()
        cur_prompt = qs

        if 'image' in line:
            image_file = line["image"]
            image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = process_images([image], image_processor, model.config)[0]
            images = image_tensor.unsqueeze(0).half().cuda()
            image_sizes = [image.size]
            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            cur_prompt = '<image>' + '\n' + cur_prompt
        else:
            images = None
            image_sizes = None

        if args.single_pred_prompt:
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
            cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        if args.benchmark:
            _sync()
            t_gen0 = time.perf_counter()
        with torch.inference_mode():
            output_ids, visual_token_num = model.generate(
                input_ids,
                images=images,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=1024,
                use_cache=True,
            )
            data_bar.set_postfix({"visual_token_num": visual_token_num})
        if args.benchmark:
            _sync()
            gen_s = time.perf_counter() - t_gen0
            gen_latencies.append(gen_s)
            tflops_meter.update(
                estimate_llava15_inference_tflops(
                    model,
                    input_ids,
                    output_ids,
                    visual_token_num,
                    IMAGE_TOKEN_INDEX,
                ),
                gen_s,
            )

        if args.save_prune_vis_dir and images is not None:
            raw_mask = getattr(model, "_vispruner_last_index_mask", None)
            if raw_mask is not None:
                try:
                    tower = model.get_vision_tower()
                    n_side = tower.num_patches_per_side
                    out_pil = render_discarded_patches_on_original(
                        image,
                        raw_mask[0],
                        n_side,
                        image_processor,
                        args.prune_vis_discarded_blend,
                    )
                    os.makedirs(args.save_prune_vis_dir, exist_ok=True)
                    safe_id = str(idx).replace("/", "_")
                    out_pil.save(os.path.join(args.save_prune_vis_dir, f"{safe_id}.png"))
                except ValueError as e:
                    if not prune_vis_warned:
                        print(f"[vispruner] prune 可视化已跳过: {e}", flush=True)
                        prune_vis_warned = True

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

    if args.benchmark:
        n = len(questions)
        e2e_s = time.perf_counter() - e2e_t0
        gen_sum = sum(gen_latencies)
        # 平均延迟（ms/条）、平均吞吐（samples/s）
        avg_lat_gen_ms = 1000.0 * gen_sum / n if n else 0.0
        thr_gen_sps = n / gen_sum if gen_sum > 0 else 0.0
        avg_lat_e2e_ms = 1000.0 * e2e_s / n if n else 0.0
        thr_e2e_sps = n / e2e_s if e2e_s > 0 else 0.0
        print(
            "\n[SQA benchmark] "
            f"samples={n} | load_time={load_s:.3f}s\n"
            f"  generate_only (model.generate): "
            f"avg_latency={avg_lat_gen_ms:.2f} ms/sample, avg_throughput={thr_gen_sps:.3f} samples/s\n"
            f"  e2e_after_load (读图/预处理/tokenize+generate): "
            f"avg_latency={avg_lat_e2e_ms:.2f} ms/sample, avg_throughput={thr_e2e_sps:.3f} samples/s\n"
            f"{tflops_meter.format_summary(e2e_s)}",
            flush=True,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--visual_token_num", type=int, default=576)
    parser.add_argument("--important_ratio", type=float, default=0.5)
    parser.add_argument(
        "--mm-vispruner-spatial-sigma",
        type=float,
        default=2.0,
        help="VisPruner 空间衰减高斯核 sigma，写入 model.config",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="打印加载时间、纯 generate/e2e 延迟吞吐，并估算 LLaVA1.5 推理 TFLOPs 与 TFLOPs/s",
    )
    parser.add_argument(
        "--save-prune-vis-dir",
        type=str,
        default=None,
        help="若设置，将对含图样本保存「丢弃 patch 淡化」后的原图可视化 PNG（文件名=question id）",
    )
    parser.add_argument(
        "--prune-vis-discarded-blend",
        type=float,
        default=0.35,
        help="丢弃区域与原图混合系数：out=orig*blend+255*(1-blend)，越小越淡（默认 0.35）",
    )
    args = parser.parse_args()

    eval_model(args)
