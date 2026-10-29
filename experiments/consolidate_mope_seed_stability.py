from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from datetime import datetime, timezone

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


RUN_ROOT = ROOT / "results" / "result7_mope_seed_stability_fdg_pooled_reference_v1_runs"
RUN_ID = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
OUTPUT_DIR = (
    ROOT / "results" / "result7_mope_seed_stability_fdg_pooled_reference_v1_consolidated" / RUN_ID
)
SEEDS = [42, 43, 44, 45, 46]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolidate validated FDG pooled-reference seed runs."
    )
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    prediction_frames = []
    kernel_frames = []
    metric_frames = []
    run_frames = []
    parameters = {}
    run_metadata = []
    common_config = None

    for seed in SEEDS:
        run_dir = args.run_root / f"seed_{seed}"
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
        run_metadata.append(json.loads((run_dir / required[4]).read_text(encoding="utf-8")))

        source_checkpoint = run_dir / "checkpoints" / f"mope_seed_{seed}.pth"
        checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
        mope.validate_checkpoint(checkpoint)
        config = checkpoint["config"]
        if config["seed"] != seed:
            raise ValueError(f"Checkpoint seed differs from run directory: {run_dir}")
        shared_keys = (
            "model_schema", "normalization_reference", "epochs", "batch_size",
            "learning_rate", "weight_decay", "ranking_margin", "aggregation_lambda",
            "aggregation_reference_center", "cellpose_diameter",
        )
        shared_config = {key: config[key] for key in shared_keys}
        if common_config is None:
            common_config = shared_config
        elif shared_config != common_config:
            raise ValueError("Cannot consolidate seed runs with different model protocols.")
        for key in ("epochs", "batch_size"):
            if run_metadata[-1][key] != config[key]:
                raise ValueError(f"Run metadata and checkpoint {key} differ: {run_dir}")
        if run_metadata[-1]["dynamic_threshold_lambda"] != config["aggregation_lambda"]:
            raise ValueError(f"Run metadata and checkpoint aggregation differ: {run_dir}")
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

    predictions.to_csv(args.output_dir / "per_plot_predictions_by_seed.csv", index=False)
    kernels.to_csv(args.output_dir / "per_kernel_scores_by_seed.csv", index=False)
    metrics.to_csv(args.output_dir / "per_seed_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "seed_summary_metrics.csv", index=False)
    runs.to_csv(args.output_dir / "training_runs.csv", index=False)
    parameter_stability.to_csv(args.output_dir / "parameter_stability.csv", index=False)
    score_correlation.to_csv(args.output_dir / "pairwise_score_spearman_correlations.csv")
    pd.DataFrame(gate_rows).to_csv(args.output_dir / "gate_stability.csv", index=False)
    make_figure(metrics, args.output_dir / "mope_seed_stability.png")

    metadata = {
        "purpose": "Full-protocol random-initialization stability analysis.",
        "seeds": SEEDS,
        "config": common_config,
        "epochs": common_config["epochs"],
        "batch_size": common_config["batch_size"],
        "learning_rate": common_config["learning_rate"],
        "weight_decay": common_config["weight_decay"],
        "ranking_margin": common_config["ranking_margin"],
        "dynamic_threshold_lambda": common_config["aggregation_lambda"],
        "n_healthy_valid_features": run_metadata[0]["n_healthy_valid_features"],
        "n_diseased_valid_features": run_metadata[0]["n_diseased_valid_features"],
        "n_test_plots": len(predictions) // len(SEEDS),
        "n_test_kernels": len(kernels) // len(SEEDS),
        "source_run_root": str(args.run_root),
        "note": "The 80 evaluation labels were not used for fitting or threshold calibration.",
    }
    (args.output_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Merged {common_config['epochs']}-epoch seed experiment into {args.output_dir}")


if __name__ == "__main__":
    main()
