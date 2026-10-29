# Reproducibility guide

This guide assumes that the required image data, kernel crops, prediction tables, and optional trained checkpoints have been obtained separately and placed according to `data/README.md`.

The current core uses schema `fdg_pooled_reference_v1`: pooled training-kernel
z-scores, three groupwise mean absolute deviations, one positive temperature,
16 training epochs and seed 42. The image threshold is class-balanced with
coefficient 0.25. Legacy checkpoint and generated-table files are not evidence
of results from this corrected implementation. Recreate the checkpoint and
predictions, then rerun the analyses before attaching numerical claims to this
version. Do not rename or relabel an old checkpoint to bypass schema checks.

## 1. Core MoPE inference

```bash
python few_shot_cellpose_mope_ranking.py --images path/to/image1.jpg path/to/image2.jpg
```

To retrain the MoPE ranker from `examples/Healthy/` and `examples/Diseased/`:

```bash
python few_shot_cellpose_mope_ranking.py --retrain
```

The default corrected-model directory is
`output_results/MoPE_Ranking_FDG_PooledReference_v1/`. Use `--output-dir` to
select another new directory without overwriting earlier runs.

## 2. Rebuild the current model workflow

```bash
python scripts/run_newdata_only_seed_models.py
```

This expects the seed image folders under `data/seeds_png/` and writes each run
to a new timestamped directory under `output_results/fdg_pooled_reference_v1/`.
The run summary records its path. For subsequent scripts, pass that run's
`training_examples/` path via `--training-root`; do not silently reuse earlier
training outputs.

## 3. Generalization analysis

```bash
python experiments/experiment_generalization_existing_data.py
```

This evaluates the independent 80-plot prediction table on:

- all evaluation samples;
- the 2024 cross-year subset;
- unseen genotypes;
- the strict 2024 + unseen-genotype subset;
- the 2025 + unseen-genotype subset.

## 4. Method comparison

```bash
python experiments/experiment_method_performance_comparison.py
```

The comparison includes MoPE, Random Forest, RankSVM, Single-Linear, ResNet18, ViT, and SVM baselines when the required inputs are present.

## 5. Threshold sensitivity

```bash
python experiments/experiment_threshold_sensitivity.py
```

This evaluates the aggregation threshold parameter over the prespecified lambda grid.

## 6. Runtime uncertainty

```bash
python experiments/experiment_runtime_uncertainty.py
```

This summarizes runtime and GPU-memory variation from stored per-image segmentation benchmark outputs.

## 7. Figure generation

```bash
python scripts/figures/plot_method_comparison_overlay_with_ranksvm.py
python scripts/figures/make_line_year_mope_plot.py
```

Figure scripts expect the corresponding result tables to exist under `results/`.

## 8. CPU implementation checks

```bash
python tests/test_fdg_gating.py
```

These synthetic checks require the standard numerical dependencies but no image
dataset, Cellpose download or GPU. They cover the gate, training-reference
normalization, aggregation, saved configuration and checkpoint rejection/round
trip. They do not reproduce the evaluation metrics.

`run_mope_seed_stability_300_parallel.py` is retained as an optional 300-epoch
diagnostic. It is not the manuscript's fixed 16-epoch training protocol and
writes separately labelled diagnostic outputs.
