from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import few_shot_cellpose_mope_ranking as mope
from experiment_mope_seed_stability import (
    EXPERT_NAMES,
    cosine_similarity,
    make_figure,
    parameter_vectors,
    summarize_seed_metrics,
)


RUN_ROOT = ROOT / "results" / "result7_mope_seed_stability_300_runs"
OUTPUT_DIR = ROOT / "results" / "result7_mope_seed_stability_300"
SEEDS = [42, 43, 44, 45, 46]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = OUTPUT_DIR / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    prediction_frames = []
    kernel_frames = []
    metric_frames = []
    run_frames = []
    parameters = {}

    for seed in SEEDS:
        run_dir = RUN_ROOT / f"seed_{seed}"
        required = [
            "per_plot_predictions_by_seed.csv",
            "per_kernel_scores_by_seed.csv",
            "per_seed_metrics.csv",
            "training_runs.csv",
            "experiment_metadata.json",
        ]
        missing = [name for name in required if not (run_dir / name).exists()]
        if missing:
            raise FileNotFoundError(f"Seed {seed} is incomplete: {missing}")
        prediction_frames.append(pd.read_csv(run_dir / required[0]))
        kernel_frames.append(pd.read_csv(run_dir / required[1]))
        metric_frames.append(pd.read_csv(run_dir / required[2]))
        run_frames.append(pd.read_csv(run_dir / required[3]))

        source_checkpoint = run_dir / "checkpoints" / f"mope_seed_{seed}.pth"
        checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
        model = mope.MoPERanker()
        model.load_state_dict(checkpoint["model_state_dict"])
        parameters[seed] = parameter_vectors(model)
        shutil.copy2(source_checkpoint, checkpoint_dir / source_checkpoint.name)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    kernels = pd.concat(kernel_frames, ignore_index=True)
    metrics = pd.concat(metric_frames, ignore_index=True)
    runs = pd.concat(run_frames, ignore_index=True)
    if len(predictions) != 400 or len(kernels) != 5 * 21835:
        raise ValueError(
            f"Unexpected merged row counts: plots={len(predictions)}, kernels={len(kernels)}"
        )
    summary = summarize_seed_metrics(metrics)

    pair_rows = []
    for index, seed_a in enumerate(SEEDS):
        for seed_b in SEEDS[index + 1 :]:
            for block in parameters[seed_a]:
                pair_rows.append(
                    {
                        "seed_a": seed_a,
                        "seed_b": seed_b,
                        "parameter_block": block,
                        "cosine_similarity": cosine_similarity(
                            parameters[seed_a][block], parameters[seed_b][block]
                        ),
                    }
                )
    parameter_stability = pd.DataFrame(pair_rows)

    score_wide = kernels.pivot_table(
        index=["year", "plot", "seed_id"],
        columns="training_seed",
        values="mope_score",
    )
    score_correlation = score_wide.corr(method="spearman")
    score_correlation.index.name = "seed_a"
    score_correlation.columns.name = "seed_b"

    gate_rows = []
    for expert in EXPERT_NAMES:
        wide = kernels.pivot_table(
            index=["year", "plot", "seed_id"],
            columns="training_seed",
            values=f"gate_{expert}",
        )
        gate_sd = wide.std(axis=1, ddof=1)
        gate_rows.append(
            {
                "expert": expert,
                "grand_mean_gate": float(wide.to_numpy().mean()),
                "mean_kernel_sd_across_seeds": float(gate_sd.mean()),
                "max_kernel_sd_across_seeds": float(gate_sd.max()),
            }
        )

    predictions.to_csv(OUTPUT_DIR / "per_plot_predictions_by_seed.csv", index=False)
    kernels.to_csv(OUTPUT_DIR / "per_kernel_scores_by_seed.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "per_seed_metrics.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "seed_summary_metrics.csv", index=False)
    runs.to_csv(OUTPUT_DIR / "training_runs.csv", index=False)
    parameter_stability.to_csv(OUTPUT_DIR / "parameter_stability.csv", index=False)
    score_correlation.to_csv(OUTPUT_DIR / "pairwise_score_spearman_correlations.csv")
    pd.DataFrame(gate_rows).to_csv(OUTPUT_DIR / "gate_stability.csv", index=False)
    make_figure(metrics, OUTPUT_DIR / "mope_seed_stability.png")

    metadata = {
        "purpose": "Full-protocol random-initialization stability analysis.",
        "seeds": SEEDS,
        "epochs": 300,
        "batch_size": 128,
        "learning_rate": mope.LEARNING_RATE,
        "weight_decay": mope.WEIGHT_DECAY,
        "ranking_margin": mope.RANKING_MARGIN,
        "dynamic_threshold_lambda": 0.25,
        "n_healthy_valid_features": 395,
        "n_diseased_valid_features": 384,
        "n_test_plots": 80,
        "n_test_kernels": 21835,
        "source_run_root": str(RUN_ROOT),
        "note": "The 80 evaluation labels were not used for fitting or threshold calibration.",
    }
    (OUTPUT_DIR / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Merged full 300-epoch seed experiment into {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
