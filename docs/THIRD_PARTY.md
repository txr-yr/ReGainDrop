# Third-party source and notices

| Component | Source | Notice / scope |
| --- | --- | --- |
| VisPruner | https://github.com/Theia-4869/VisPruner | Apache-2.0; base repository identified in supplied package metadata and accompanying paper. |
| LLaVA | https://github.com/haotian-liu/LLaVA | Apache-2.0; Haotian Liu and other original headers remain in `llava/`. |
| Transformers / Qwen | https://github.com/huggingface/transformers/tree/v5.11.0/src/transformers/models/qwen2_5_vl | Apache-2.0; Qwen and Hugging Face headers retained. Qwen sources match this base with custom pruning changes. |
| MMF evaluator | https://github.com/facebookresearch/mmf | BSD notice accompanies `llava/eval/m4c_evaluator.py`. |
| ToMe matching lineage | https://github.com/facebookresearch/ToMe | Acknowledged by upstream VisPruner. Exact line-level ancestry was not established from the supplied archive; consult its own license when directly reusing its code. |

The supplied archive did not include Git history or its advertised LICENSE file. The upstream Apache license has been restored. This repository does not claim independent authorship of inherited implementations. Modified files carry release notices, and existing attribution and copyright headers remain intact.

The root Apache license covers components offered under that license; it does not override separately licensed portions, checkpoints, benchmark data, APIs, or dependencies. Consult their respective upstream terms when redistributing or using them.

The original VisPruner paper is *Beyond Text-Visual Attention: Exploiting Visual Cues for Effective Token Pruning in VLMs* (Zhang et al., 2025; https://arxiv.org/abs/2412.01818).
