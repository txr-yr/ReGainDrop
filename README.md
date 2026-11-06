# ReGainDrop

Research code accompanying **Rethinking Visual Cues for Training-Free Token Pruning in Vision-Language Models**.

Xingpeng Zhong, Xirui Tian, Junhao Liu, Guanquan Zhu, and Xingyi Zhong (corresponding author).

ReGainDrop studies training-free visual token pruning using global visual relevance, relative representation gain, and spatial-similarity regularization. The manuscript evaluates LLaVA-1.5, LLaVA-NeXT, and Qwen2.5-VL.

## Release status

This is a cleaned **author-supplied development snapshot**, not a verified reproduction release. Packaging and portable paths have been updated; model computations have been preserved. The snapshot has known differences from the manuscript in the gain formula, CLIP query pooling, and complementary-token selection. Read [reproducibility notes](docs/REPRODUCIBILITY.md) before interpreting outputs or comparing with paper tables. No new benchmark results are claimed by this release.

## Start here

```bash
git clone https://github.com/txr-yr/ReGainDrop.git
cd ReGainDrop
conda create -n regaindrop-llava python=3.10 -y
conda activate regaindrop-llava
python -m pip install -e '.[eval]'
```

Linux with a CUDA-capable GPU is the intended inference environment. Model weights and datasets are downloaded separately under their own terms. Follow [installation and evaluation](docs/INSTALLATION.md) for the required data layout and commands. Qwen2.5-VL uses a **separate environment** and the version-checked overlay helper; it must not be installed as a top-level `qwen2_5_vl` Python package.

## Layout

- `llava/`: LLaVA model integration and evaluation utilities.
- `scripts/v1_5/eval/`, `scripts/v1_6/eval/`: benchmark launchers.
- `qwen2_5_vl/`: author-modified Transformers 5.11.0 modules; experimental overlay sources.
- `tools/prepare_qwen_overlay.py`: creates a separate patched Transformers tree without overwriting an installed package.
- `docs/REPRODUCIBILITY.md`: implementation differences and validation scope.

## Citation and contact

Please cite the manuscript as an unpublished manuscript until publication details are available. No journal acceptance or DOI is claimed. Author metadata is provided in [CITATION.cff](CITATION.cff).

Correspondence: Xingyi Zhong, School of Creative and Design, Guangdong Construction Polytechnic, Guangzhou 510440, China. Email: zhongxingyi@gdcvi.edu.cn.

## Acknowledgments and licenses

This project builds on [VisPruner](https://github.com/Theia-4869/VisPruner), [LLaVA](https://github.com/haotian-liu/LLaVA), and [Hugging Face Transformers](https://github.com/huggingface/transformers). Reused source notices are retained; the MMF evaluator and matching/pruning lineage have separate notices. See [NOTICE](NOTICE) and [third-party attribution](docs/THIRD_PARTY.md). The root Apache-2.0 license does not override file-specific terms or model/dataset licenses.
