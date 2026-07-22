from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import few_shot_cellpose_mope_ranking as mope
from experiment_generalization_existing_data import build_subsets, calculate_metrics


DEFAULT_KERNEL_CSV = (
    ROOT / "results" / "result2" / "per_kernel" / "table8_per_kernel_predictions.csv"
)
DEFAULT_PLOT_CSV = (
    ROOT / "results" / "result2" / "per_plot" / "table8_per_plot_predictions.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "results" / "result7_mope_seed_stability"
DEFAULT_TRAINING_ROOT = (
    ROOT / "output_results" / "current_seed_models" / "training_examples"
)
METRICS = ["pearson_r", "spearman_rho", "mae", "rmse", "bias"]
EXPERT_NAMES = ["color", "texture", "morphology"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrain MoPE with multiple seeds and quantify result stability."
    )
    parser.add_argument("--kernel-csv", type=Path, default=DEFAULT_KERNEL_CSV)
    parser.add_argument("--plot-csv", type=Path, default=DEFAULT_PLOT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--training-root", type=Path, default=DEFAULT_TRAINING_ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=mope.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=mope.BATCH_SIZE)
    parser.add_argument("--lambda-value", type=float, default=0.25)
    parser.add_argument("--training-year", type=int, default=2025)
    parser.add_argument("--training-line", default="MN-Rothsay")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def score_model(
    model: mope.MoPERanker, features: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        scores, gates, experts = model(torch.as_tensor(features, dtype=torch.float32))
    return (
        scores.squeeze(1).cpu().numpy(),
        gates.cpu().numpy(),
        experts.cpu().numpy(),
    )


def load_data(kernel_path: Path, plot_path: Path, training_root: Path):
    kernel_df = pd.read_csv(kernel_path)
    plot_df = pd.read_csv(plot_path)
    healthy_raw, feature_names = mope.extract_features_from_dir(
        str(training_root / "Healthy")
    )
    diseased_raw, diseased_names = mope.extract_features_from_dir(
        str(training_root / "Diseased")
    )
    if list(feature_names) != list(diseased_names):
        raise ValueError("Healthy and diseased training feature names differ.")
    missing = set(feature_names).difference(kernel_df.columns)
    if missing:
        raise ValueError(f"Kernel table is missing features: {sorted(missing)}")
    h_norm, d_norm, feat_mean, feat_std = mope.normalize_features(
        healthy_raw, diseased_raw
    )
    test_norm = (
        kernel_df[list(feature_names)].to_numpy(dtype=np.float64) - feat_mean
    ) / feat_std
    return kernel_df, plot_df, h_norm, d_norm, test_norm, feature_names, feat_mean, feat_std


def aggregate_plot_predictions(
    kernel_df: pd.DataFrame,
    scores: np.ndarray,
    healthy_mean: float,
    diseased_mean: float,
    lambda_value: float,
) -> pd.DataFrame:
    work = kernel_df[["year", "plot"]].copy()
    work["mope_score"] = scores
    reference_mean = (healthy_mean + diseased_mean) / 2.0
    rows = []
    for (year, plot), group in work.groupby(["year", "plot"], sort=False):
        values = group["mope_score"].to_numpy(dtype=float)
        threshold = diseased_mean + lambda_value * (values.mean() - reference_mean)
        rows.append(
            {
                "year": int(year),
                "plot": str(plot),
                "n_kernels": int(len(values)),
                "mope_mean_score": float(values.mean()),
                "mope_vsk_threshold": float(threshold),
                "mope_predicted_vsk_pct": float(np.mean(values >= threshold) * 100.0),
            }
        )
    return pd.DataFrame(rows)


def parameter_vectors(model: mope.MoPERanker) -> dict[str, np.ndarray]:
    state = model.state_dict()
    prefixes = {
        "color_expert": "expert_color.fc.",
        "texture_expert": "expert_texture.fc.",
        "morphology_expert": "expert_morph.fc.",
        "gating": "gating.fc.",
    }
    vectors = {}
    for name, prefix in prefixes.items():
        values = [
            tensor.detach().cpu().numpy().ravel()
            for key, tensor in state.items()
            if key.startswith(prefix)
        ]
        vectors[name] = np.concatenate(values)
    return vectors


def summarize_seed_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (subset, metric), group in metrics.groupby(["subset", "metric"], sort=False):
        values = group["estimate"].to_numpy(dtype=float)
        mean = float(values.mean())
        sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        half = float(t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values))) if len(values) > 1 else 0.0
        rows.append(
            {
                "subset": subset,
                "metric": metric,
                "n_seeds": len(values),
                "mean": mean,
                "sd": sd,
                "ci_lower": mean - half,
                "ci_upper": mean + half,
            }
        )
    return pd.DataFrame(rows)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else np.nan


def make_figure(metrics: pd.DataFrame, output_path: Path) -> None:
    subsets = list(metrics["subset"].drop_duplicates())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, metric, label in zip(axes, ["mae", "rmse"], ["MAE", "RMSE"], strict=True):
        selected = metrics.loc[metrics["metric"] == metric]
        for seed, group in selected.groupby("seed"):
            ordered = group.set_index("subset").loc[subsets]
            ax.plot(range(len(subsets)), ordered["estimate"], marker="o", alpha=0.75, label=f"Seed {seed}")
        ax.set_xticks(range(len(subsets)), [s.replace("_", "\n") for s in subsets], rotation=0)
        ax.set_ylabel(f"{label} (percentage points)")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("MoPE random-seed stability across evaluation subsets")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    (
        kernel_df,
        plot_df,
        h_norm,
        d_norm,
        test_norm,
        feature_names,
        feat_mean,
        feat_std,
    ) = load_data(args.kernel_csv, args.plot_csv, args.training_root)

    metric_rows = []
    prediction_frames = []
    kernel_frames = []
    run_rows = []
    parameter_by_seed: dict[int, dict[str, np.ndarray]] = {}
    subsets = build_subsets(plot_df, args.training_year, args.training_line)

    for seed in args.seeds:
        set_seed(seed)
        model = mope.MoPERanker()
        trainer = mope.PairwiseRankingTrainer(
            model,
            lr=mope.LEARNING_RATE,
            weight_decay=mope.WEIGHT_DECAY,
            margin=mope.RANKING_MARGIN,
        )
        started = time.perf_counter()
        history = trainer.train(
            h_norm,
            d_norm,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=True,
            silent=True,
        )
        elapsed = time.perf_counter() - started
        healthy_scores, _, _ = score_model(model, h_norm)
        diseased_scores, _, _ = score_model(model, d_norm)
        scores, gates, experts = score_model(model, test_norm)
        h_mean = float(healthy_scores.mean())
        d_mean = float(diseased_scores.mean())
        predictions = aggregate_plot_predictions(
            kernel_df, scores, h_mean, d_mean, args.lambda_value
        )
        metadata = plot_df[["year", "plot", "line", "manual_vsk_pct"]].copy()
        metadata["plot"] = metadata["plot"].astype(str)
        predictions["plot"] = predictions["plot"].astype(str)
        predictions = metadata.merge(predictions, on=["year", "plot"], validate="one_to_one")
        predictions.insert(0, "seed", seed)
        prediction_frames.append(predictions)

        kernel_seed = kernel_df[["year", "plot", "seed_id"]].copy()
        kernel_seed.insert(0, "training_seed", seed)
        kernel_seed["mope_score"] = scores
        for index, name in enumerate(EXPERT_NAMES):
            kernel_seed[f"gate_{name}"] = gates[:, index]
            kernel_seed[f"expert_{name}_score"] = experts[:, index]
        kernel_frames.append(kernel_seed)

        for subset_name, mask in subsets.items():
            selected = predictions.loc[mask]
            values = calculate_metrics(
                selected["manual_vsk_pct"].to_numpy(dtype=float),
                selected["mope_predicted_vsk_pct"].to_numpy(dtype=float),
            )
            for metric in METRICS:
                metric_rows.append(
                    {
                        "seed": seed,
                        "subset": subset_name,
                        "n": len(selected),
                        "metric": metric,
                        "estimate": values[metric],
                    }
                )

        parameter_by_seed[seed] = parameter_vectors(model)
        checkpoint_path = checkpoint_dir / f"mope_seed_{seed}.pth"
        torch.save(
            {
                "seed": seed,
                "model_state_dict": model.state_dict(),
                "feat_mean": feat_mean,
                "feat_std": feat_std,
                "feature_names": list(feature_names),
                "history": history,
                "healthy_score_mean": h_mean,
                "diseased_score_mean": d_mean,
            },
            checkpoint_path,
        )
        run_rows.append(
            {
                "seed": seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "training_seconds": elapsed,
                "final_loss": history["loss"][-1],
                "final_pair_accuracy": history["pair_acc"][-1],
                "healthy_score_mean": h_mean,
                "diseased_score_mean": d_mean,
                "checkpoint": str(checkpoint_path),
            }
        )
        print(f"Completed seed {seed} in {elapsed:.1f} s")

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    kernels = pd.concat(kernel_frames, ignore_index=True)
    runs = pd.DataFrame(run_rows)
    summary = summarize_seed_metrics(metrics)

    pair_rows = []
    seeds = list(args.seeds)
    for i, seed_a in enumerate(seeds):
        for seed_b in seeds[i + 1 :]:
            for block in parameter_by_seed[seed_a]:
                pair_rows.append(
                    {
                        "seed_a": seed_a,
                        "seed_b": seed_b,
                        "parameter_block": block,
                        "cosine_similarity": cosine_similarity(
                            parameter_by_seed[seed_a][block], parameter_by_seed[seed_b][block]
                        ),
                    }
                )
    parameter_stability = pd.DataFrame(pair_rows)

    score_wide = kernels.pivot_table(
        index=["year", "plot", "seed_id"], columns="training_seed", values="mope_score"
    )
    correlation = score_wide.corr(method="spearman")
    correlation.index.name = "seed_a"
    correlation.columns.name = "seed_b"

    gate_rows = []
    for expert in EXPERT_NAMES:
        wide = kernels.pivot_table(
            index=["year", "plot", "seed_id"],
            columns="training_seed",
            values=f"gate_{expert}",
        )
        gate_rows.append(
            {
                "expert": expert,
                "grand_mean_gate": float(wide.to_numpy().mean()),
                "mean_kernel_sd_across_seeds": float(wide.std(axis=1, ddof=1).mean()) if len(seeds) > 1 else 0.0,
                "max_kernel_sd_across_seeds": float(wide.std(axis=1, ddof=1).max()) if len(seeds) > 1 else 0.0,
            }
        )

    predictions.to_csv(args.output_dir / "per_plot_predictions_by_seed.csv", index=False)
    kernels.to_csv(args.output_dir / "per_kernel_scores_by_seed.csv", index=False)
    metrics.to_csv(args.output_dir / "per_seed_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "seed_summary_metrics.csv", index=False)
    runs.to_csv(args.output_dir / "training_runs.csv", index=False)
    parameter_stability.to_csv(args.output_dir / "parameter_stability.csv", index=False)
    correlation.to_csv(args.output_dir / "pairwise_score_spearman_correlations.csv")
    pd.DataFrame(gate_rows).to_csv(args.output_dir / "gate_stability.csv", index=False)
    make_figure(metrics, args.output_dir / "mope_seed_stability.png")

    metadata = {
        "purpose": "Random-initialization stability of MoPE under the published training protocol.",
        "seeds": seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": mope.LEARNING_RATE,
        "weight_decay": mope.WEIGHT_DECAY,
        "ranking_margin": mope.RANKING_MARGIN,
        "dynamic_threshold_lambda": args.lambda_value,
        "n_healthy_valid_features": len(h_norm),
        "n_diseased_valid_features": len(d_norm),
        "n_test_kernels": len(test_norm),
        "note": "The 80 evaluation labels are never used for fitting or threshold calibration.",
    }
    (args.output_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Wrote multi-seed stability outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
