# Reproducibility status

This release preserves the computations in the author-supplied `ours.zip`. The manuscript remains authoritative for the stated method and reported results. Repository cleanup is not a rerun of the experiments, and no new results, training, or benchmark scores are supplied.

## Known differences requiring reconciliation

1. `llava/model/multimodal_encoder/clip_encoder.py` computes a ratio of output and input feature norms. Equation (7) in the manuscript instead uses the norm of their difference, divided by the input norm plus epsilon.
2. The CLIP global query/key aggregation includes CLS before selecting patch columns. The manuscript defines patch-only pooling. Its fallback averages attention maps, which is not equivalent to applying softmax to the pooled-query logits.
3. `qwen2_5_vl/modeling_qwen2_5_vl.py` uses the same output/input norm ratio instead of the manuscript's residual-gain expression.
4. The complementary-token loops in `llava/model/llava_arch.py` and the Qwen module preserve score ordering and repeatedly prune the same interleaved group. Manuscript Algorithm 1 alternates the active group and sorts surviving original indices each round.
5. The implementation normalizes grid distances by `sqrt(H^2+W^2)`. The manuscript does not fully specify the coordinate scale; preserve this convention when interpreting sigma.

Correcting items 1-4 changes numerical behavior. This cleanup intentionally does not silently rewrite the method or claim that revised outputs reproduce the paper tables. An author-validated matching implementation/configuration is still needed for a reproducibility release.

## Environment and data caveats

- LLaVA uses Transformers 4.37.2; Qwen overlay sources correspond to Transformers 5.11.0. Use separate environments.
- Qwen's modules use internal relative imports and are overlay source files, not a standalone importable top-level package. Qwen end-to-end pruning/generation has not been tested during release preparation. Regenerating the modular source would erase custom modeling changes.
- LLaVA-1.5 POPE launcher defaults to `popular`, not all three POPE subsets. Set `POPE_CATEGORY=all` explicitly for the combined input. Do not equate a default run with a full paper aggregate.
- ScienceQA conversion defaults to the FULL split; use `--hf-config ScienceQA-IMG` for the image subset described in the paper.
- Original launch flags with `vispruner` in their names remain for compatibility with checkpoint/configuration fields; renaming them blindly would break execution.
- Checkpoints, licensed data, official benchmark helper files, and detailed GPU runtime records are not included. Some benchmark launchers require those separately downloaded helpers.

## What was checked

Release preparation checks Python syntax, retained algorithm AST equivalence, repository contents, portable launcher paths, and license/attribution retention. This does not establish numerical equivalence to the manuscript, successful CUDA execution, or the reproducibility of any table. See the delivered validation report for the checks actually completed.
