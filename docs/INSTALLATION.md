# Installation and evaluation

Read [reproducibility status](REPRODUCIBILITY.md) first. These instructions describe the cleaned development snapshot; full GPU inference was not rerun for this release.

## LLaVA environment

Use Linux, Bash, an NVIDIA GPU and Python 3.10 in an isolated environment:

```bash
conda create -n regaindrop-llava python=3.10 -y
conda activate regaindrop-llava
python -m pip install --upgrade pip
python -m pip install -e '.[eval]'
```

The project retains PyTorch 2.1.2 / torchvision 0.16.2 and Transformers 4.37.2, with NumPy below 2 and PEFT 0.7.1 for compatibility with that stack. CUDA wheels/drivers must match your platform. FlashAttention is optional; select a build compatible with the pinned Torch/CUDA environment. Quantized loading requires the `quantization` extra. The inherited serving/training helpers are not a tested public service and are not needed for the evaluation path.

Download official LLaVA checkpoints into `models/llava-v1.5-7b` and/or `models/llava-v1.6-vicuna-7b`, or set `CKPT_DIR` and `CKPT` to your own path/name. The checkpoint directory must include `config.json` and the required model files.

## POPE and ScienceQA preparation

Converters accept locally downloaded Hugging Face parquet files. Inspect each converter's `--help` for the selected schema; no dataset is downloaded automatically by the repository.

```bash
python scripts/prepare_hf_pope_for_llava_eval.py --help
python scripts/prepare_hf_sqa_for_llava_eval.py --help
```

The POPE launcher expects `DATA_DIR/pope/val2014` for images and `DATA_DIR/pope/coco` for labels. Its default `DATA_DIR` is `datasets/pope`, so the nested path is deliberate and matches the supplied converter. Default questions are `playground/data/eval/pope/llava_pope_test.jsonl`; override with `QUESTION_FILE`.

ScienceQA expects `DATA_DIR/scienceqa/{problems.json,pid_splits.json,images/test/}`. Its default `DATA_DIR` is `datasets/sqa`, and the questions are `playground/data/eval/scienceqa/llava_test_CQM-I.json`. Use the converter's `--hf-config ScienceQA-IMG` for the paper's image subset. The LLaVA-1.5 launcher defaults to `CKPT=llava-v1.5-7b`; override it to match your downloaded checkpoint directory.

Example invocations after data/model preparation:

```bash
CKPT=llava-v1.5-7b POPE_CATEGORY=all POPE_SAVE_PRUNE_VIS=0 \
  CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/pope.sh 64 0.5
CKPT=llava-v1.5-7b SQA_SAVE_PRUNE_VIS=0 \
  CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/sqa.sh 64 0.5
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_6/eval/pope.sh 64 0.5
```

The two positional arguments are the per-view visual token budget and important-token ratio. LLaVA-NeXT total retained tokens depend on its views. These commands are usage examples, not verified reproductions of table entries. Optional `POPE_SPATIAL_SIGMA` / `SQA_SPATIAL_SIGMA` override the snapshot's sigma. Check the launcher's current defaults before comparing outputs.

## Other benchmarks

Launchers are retained for VQAv2, TextVQA, MME, MMBench, MMBench-CN, MM-Vet, GQA and VizWiz. Configure `CKPT_DIR`, `DATA_DIR`, and benchmark question files. External annotations/evaluation helpers expected under `playground/data/eval` are not included in the snapshot. Follow [LLaVA evaluation instructions](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md) and the benchmark providers; inspect each script before running. Do not interpret the presence of inherited launchers as a claim that all those benchmarks appear in this manuscript.

## Experimental Qwen2.5-VL overlay

Use a separate Python 3.11 environment. Install a compatible PyTorch/torchvision pair for your CUDA version, then:

```bash
python -m pip install -r requirements-qwen.txt
python tools/prepare_qwen_overlay.py --destination .qwen_overlay
export PYTHONPATH="$PWD/.qwen_overlay${PYTHONPATH:+:$PYTHONPATH}"
python -c "import transformers; print(transformers.__version__, transformers.__file__)"
```

The helper requires Transformers 5.11.0, checks the two original source hashes, creates a new local copy of its Python package, and replaces only the two custom configuration/modeling modules. It refuses an existing destination and does not modify site-packages. Use a fresh interpreter after changing `PYTHONPATH`. The imported path should point inside `.qwen_overlay/transformers`.

Qwen's pruning options are in `qwen2_5_vl/configuration_qwen2_5_vl.py`. This snapshot has no validated complete Qwen benchmark runner; the overlay alone is not a reproduction recipe. A model-specific evaluation harness and author-validated configuration are still required.
