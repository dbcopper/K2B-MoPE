# K2B-MoPE

K2B-MoPE is a kernel-to-batch wheat Fusarium head blight (FHB) severity assessment workflow. It extracts individual wheat kernels from dense RGB images, scores each kernel with a phenotype-aware Mixture of Phenotypic Experts (MoPE) ranking model, and aggregates kernel-level evidence into sample-level visual scabby kernel (VSK) severity estimates.

This repository contains the code release accompanying the manuscript:

> K2B-MoPE: An Efficient Mixture of Phenotypic Experts Framework for Interpretable Kernel-to-Batch Fusarium Head Blight Severity Assessment in Wheat

## What is included

- `few_shot_cellpose_mope_ranking.py`: core SeedPose/Cellpose + MoPE ranking pipeline.
- `scripts/`: data conversion, model training, and figure scripts.
- `experiments/`: method comparison, generalization, threshold sensitivity, runtime uncertainty, and seed-stability diagnostics.
- `data/README.md`: expected data layout and table schemas.
- `docs/REPRODUCIBILITY.md`: commands used to reproduce the manuscript-side analyses once data/results are available.

Raw images, private fieldbooks, trained checkpoints, generated result folders, and manuscript files are intentionally not included.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If you use GPU acceleration, install the PyTorch build that matches your CUDA version before installing the remaining requirements.

## Minimal usage

Train or load a MoPE model and score images:

```bash
python few_shot_cellpose_mope_ranking.py --images path/to/image1.jpg path/to/image2.jpg
```

Force retraining from `examples/Healthy/` and `examples/Diseased/`:

```bash
python few_shot_cellpose_mope_ranking.py --retrain
```

Run the main model rebuild workflow if the expected `data/seeds_png/` folders are available:

```bash
python scripts/run_newdata_only_seed_models.py
```

## Manuscript analysis scripts

These scripts expect the prediction tables described in `data/README.md`.

```bash
python experiments/experiment_generalization_existing_data.py
python experiments/experiment_method_performance_comparison.py
python experiments/experiment_threshold_sensitivity.py
python experiments/experiment_runtime_uncertainty.py
python scripts/figures/plot_method_comparison_overlay_with_ranksvm.py
```

## Data availability

The code is structured so that data can be released separately from the source code. Place released data under the paths documented in `data/README.md`.

The key independent evaluation table used by the manuscript contains 80 plot-level samples: 31 from 2024 and 49 from 2025.

## Reproducibility notes

- The model uses fixed default training hyperparameters in `few_shot_cellpose_mope_ranking.py`.
- The manuscript-side subset/generalization analysis uses existing prediction tables and does not require new data acquisition.
- Some diagnostic scripts are intentionally included even when they do not support a positive manuscript claim, so readers can audit robustness decisions.

## License

This project is released under the MIT License. See `LICENSE`.
