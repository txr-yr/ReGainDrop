# ReGainDrop release preparation (2026-11-02): distribution notice added.
# Implementation retained from the author-supplied development snapshot; see NOTICE.
import argparse
import json
import math
import os
import time

import shortuuid
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.eval.tflops import TFlopsAccumulator, estimate_llava15_inference_tflops
from llava.eval.vispruner_prune_vis import render_discarded_patches_on_original
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.utils import disable_torch_init


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, conv_mode):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, conv_mode, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, conv_mode)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


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
    if getattr(args, "save_vit_heatmap_layer", None) is not None:
        model._vit_heatmap_layer = int(args.save_vit_heatmap_layer)
    else:
        model._vit_heatmap_layer = getattr(model.config, "mm_vision_heatmap_layer", None)
    if getattr(args, "save_vit_heatmap_cls_dir", None):
        os.makedirs(args.save_vit_heatmap_cls_dir, exist_ok=True)
    if getattr(args, "save_vit_heatmap_vtt_dir", None):
        os.makedirs(args.save_vit_heatmap_vtt_dir, exist_ok=True)

    # Data
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    if args.pope_categories:
        allowed = {c.strip() for c in args.pope_categories.split(",") if c.strip()}
        questions = [q for q in questions if q.get("category") in allowed]
        if not questions:
            raise ValueError(
                f"--pope-categories {args.pope_categories!r} 过滤后无样本，请检查 question-file 与类别名"
            )
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(
        questions, args.image_folder, tokenizer, image_processor, model.config, args.conv_mode
    )

    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    gen_latencies = []
    tflops_meter = TFlopsAccumulator()
    e2e_t0 = time.perf_counter()
    prune_vis_warned = False

    data_bar = tqdm(zip(data_loader, questions), total=len(questions))
    for (input_ids, image_tensors, image_sizes), line in data_bar:
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        image_tensors = image_tensors.to(dtype=torch.float16, device='cuda', non_blocking=True)

        if args.save_vit_heatmap_cls_dir or args.save_vit_heatmap_vtt_dir:
            model._vit_heatmap_cls_dir = args.save_vit_heatmap_cls_dir
            model._vit_heatmap_vtt_dir = args.save_vit_heatmap_vtt_dir
            model._vit_heatmap_question_id = str(idx).replace("/", "_")
        else:
            model._vit_heatmap_cls_dir = None
            model._vit_heatmap_vtt_dir = None
            model._vit_heatmap_question_id = None

        if args.benchmark:
            _sync()
            t_gen0 = time.perf_counter()
        with torch.inference_mode():
            output_ids, visual_token_num = model.generate(
                input_ids,
                images=image_tensors,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)
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
                    num_beams=args.num_beams,
                ),
                gen_s,
            )

        if args.save_prune_vis_dir:
            raw_mask = getattr(model, "_vispruner_last_index_mask", None)
            if raw_mask is not None:
                try:
                    pil = Image.open(os.path.join(args.image_folder, line["image"])).convert("RGB")
                    tower = model.get_vision_tower()
                    n_side = tower.num_patches_per_side
                    out_pil = render_discarded_patches_on_original(
                        pil,
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
        avg_lat_gen_ms = 1000.0 * gen_sum / n if n else 0.0
        thr_gen_sps = n / gen_sum if gen_sum > 0 else 0.0
        avg_lat_e2e_ms = 1000.0 * e2e_s / n if n else 0.0
        thr_e2e_sps = n / e2e_s if e2e_s > 0 else 0.0
        print(
            "\n[POPE benchmark] "
            f"samples={n} | load_time={load_s:.3f}s\n"
            f"  generate_only (model.generate): "
            f"avg_latency={avg_lat_gen_ms:.2f} ms/sample, avg_throughput={thr_gen_sps:.3f} samples/s\n"
            f"  e2e_after_load (DataLoader+generate): "
            f"avg_latency={avg_lat_e2e_ms:.2f} ms/sample, avg_throughput={thr_e2e_sps:.3f} samples/s\n"
            f"{tflops_meter.format_summary(e2e_s)}",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--visual_token_num", type=int, default=576)
    parser.add_argument("--important_ratio", type=float, default=0.5)
    parser.add_argument(
        "--mm-vispruner-spatial-sigma",
        type=float,
        default=2.0,
        help="VisPruner 相似度×空间高斯核：mm_vispruner_spatial_sigma，推理时写入 model.config",
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
        help="若设置，保存丢弃 patch 淡化后的可视化 PNG（文件名=question_id）",
    )
    parser.add_argument(
        "--prune-vis-discarded-blend",
        type=float,
        default=0.35,
        help="丢弃区域 blend：out=orig*blend+255*(1-blend)",
    )
    parser.add_argument(
        "--pope-categories",
        type=str,
        default=None,
        help="仅推理 POPE 中这些类别，逗号分隔，如 popular 或 popular,random,adversarial",
    )
    parser.add_argument(
        "--save-vit-heatmap-layer",
        type=int,
        default=None,
        help="仅热力图：ViT encoder 层下标(0=第一层, -1=最后一层)。不设则用 config 的 mm_vision_heatmap_layer，再否则与 mm_vision_select_layer(剪枝层) 相同",
    )
    parser.add_argument(
        "--save-vit-heatmap-cls-dir",
        type=str,
        default=None,
        help="若设置，在 --save-vit-heatmap-layer 指定层(见上)保存各 patch 的 CLS 注意力热力图 PNG",
    )
    parser.add_argument(
        "--save-vit-heatmap-vtt-dir",
        type=str,
        default=None,
        help="若设置，在同层保存 EUTI-VTT 分数热力图 PNG；与 CLS 分目录",
    )
    args = parser.parse_args()

    eval_model(args)
