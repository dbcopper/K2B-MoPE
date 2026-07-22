# Reproducibility guide

This guide assumes that the required image data, kernel crops, prediction tables, and optional trained checkpoints have been obtained separately and placed according to `data/README.md`.

## 1. Core MoPE inference

```bash
python few_shot_cellpose_mope_ranking.py --images path/to/image1.jpg path/to/image2.jpg
```

To retrain the MoPE ranker from `examples/Healthy/` and `examples/Diseased/`:

```bash
python few_shot_cellpose_mope_ranking.py --retrain
```

## 2. Rebuild the current model workflow

```bash
python scripts/run_newdata_only_seed_models.py
```

This expects the released seed image folders under `data/seeds_png/` and writes model outputs under `output_results/current_seed_models/`.

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
