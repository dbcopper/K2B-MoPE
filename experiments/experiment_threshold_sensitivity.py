from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from experiment_generalization_existing_data import (
    bootstrap_distributions,
    build_subsets,
    calculate_metrics,
    percentile_interval,
    two_sided_bootstrap_p,
)


DEFAULT_KERNEL_CSV = (
    ROOT / "results" / "result2" / "per_kernel" / "table8_per_kernel_predictions.csv"
)
DEFAULT_PLOT_CSV = (
    ROOT / "results" / "result2" / "per_plot" / "table8_per_plot_predictions.csv"
)
DEFAULT_THRESHOLD_INFO = ROOT / "results" / "result2" / "models" / "threshold_info.json"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "result6_threshold_sensitivity"
DEFAULT_LAMBDAS = [0.0, 0.1, 0.2, 0.25, 0.3, 0.5, 0.75, 1.0]
METRICS = ["pearson_r", "spearman_rho", "mae", "rmse", "bias"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MoPE sample-level sensitivity to the dynamic-threshold lambda."
    )
    parser.add_argument("--kernel-csv", type=Path, default=DEFAULT_KERNEL_CSV)
    parser.add_argument("--plot-csv", type=Path, default=DEFAULT_PLOT_CSV)
    parser.add_argument("--threshold-info", type=Path, default=DEFAULT_THRESHOLD_INFO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS)
    parser.add_argument("--reference-lambda", type=float, default=0.25)
    parser.add_argument("--training-year", type=int, default=2025)
    parser.add_argument("--training-line", default="MN-Rothsay")
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260703)
    return parser.parse_args()


def lambda_key(value: float) -> str:
    return "lambda_" + format(value, ".6g").replace("-", "m").replace(".", "p")


def aggregate_predictions(
    kernel_df: pd.DataFrame,
    lambdas: list[float],
    healthy_mean: float,
    diseased_mean: float,
) -> pd.DataFrame:
    reference_mean = (healthy_mean + diseased_mean) / 2.0
    rows = []
    for (year, plot), group in kernel_df.groupby(["year", "plot"], sort=False):
        scores = group["mope_score"].to_numpy(dtype=float)
        row: dict[str, object] = {
            "year": int(year),
            "plot": str(plot),
            "n_kernels": int(len(scores)),
            "mean_score": float(scores.mean()),
        }
        for value in lambdas:
            key = lambda_key(value)
            threshold = diseased_mean + value * (float(scores.mean()) - reference_mean)
            row[f"{key}_threshold"] = float(threshold)
            row[f"{key}_prediction"] = float(np.mean(scores >= threshold) * 100.0)
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate(
    merged: pd.DataFrame,
    lambdas: list[float],
    reference_lambda: float,
    training_year: int,
    training_line: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subsets = build_subsets(merged, training_year, training_line)
    rng = np.random.default_rng(seed)
    metric_rows = []
    paired_rows = []
    reference_key = lambda_key(reference_lambda)

    for subset_name, mask in subsets.items():
        selected = merged.loc[mask].reset_index(drop=True)
        y_true = selected["manual_vsk_pct"].to_numpy(dtype=float)
        predictions = {
            lambda_key(value): selected[f"{lambda_key(value)}_prediction"].to_numpy(
                dtype=float
            )
            for value in lambdas
        }
        indices = rng.integers(
            0, len(selected), size=(n_bootstrap, len(selected)), dtype=np.int64
        )
        distributions = bootstrap_distributions(y_true, predictions, indices)
        point_metrics = {
            key: calculate_metrics(y_true, values) for key, values in predictions.items()
        }
        for value in lambdas:
            key = lambda_key(value)
            for metric in METRICS:
                lower, upper, valid_count = percentile_interval(
                    distributions[key][metric]
                )
                metric_rows.append(
                    {
                        "subset": subset_name,
                        "lambda": value,
                        "n": len(selected),
                        "metric": metric,
                        "estimate": point_metrics[key][metric],
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "valid_bootstrap_resamples": valid_count,
                    }
                )

        for value in lambdas:
            key = lambda_key(value)
            if key == reference_key:
                continue
            improvements = {
                "pearson_r": distributions[reference_key]["pearson_r"]
                - distributions[key]["pearson_r"],
                "spearman_rho": distributions[reference_key]["spearman_rho"]
                - distributions[key]["spearman_rho"],
                "mae": distributions[key]["mae"] - distributions[reference_key]["mae"],
                "rmse": distributions[key]["rmse"] - distributions[reference_key]["rmse"],
                "absolute_bias": np.abs(distributions[key]["bias"])
                - np.abs(distributions[reference_key]["bias"]),
            }
            point_improvements = {
                "pearson_r": point_metrics[reference_key]["pearson_r"]
                - point_metrics[key]["pearson_r"],
                "spearman_rho": point_metrics[reference_key]["spearman_rho"]
                - point_metrics[key]["spearman_rho"],
                "mae": point_metrics[key]["mae"] - point_metrics[reference_key]["mae"],
                "rmse": point_metrics[key]["rmse"] - point_metrics[reference_key]["rmse"],
                "absolute_bias": abs(point_metrics[key]["bias"])
                - abs(point_metrics[reference_key]["bias"]),
            }
            for metric, values in improvements.items():
                lower, upper, valid_count = percentile_interval(values)
                probability, p_value = two_sided_bootstrap_p(values)
                paired_rows.append(
                    {
                        "subset": subset_name,
                        "reference_lambda": reference_lambda,
                        "comparison_lambda": value,
                        "n": len(selected),
                        "metric": metric,
                        "reference_improvement_estimate": point_improvements[metric],
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "probability_reference_better": probability,
                        "two_sided_bootstrap_p": p_value,
                        "valid_bootstrap_resamples": valid_count,
                        "interpretation": "Positive values favour the reference lambda.",
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(paired_rows)


def build_paper_table(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics.loc[metrics["subset"] == "all_evaluation"].copy()
    selected["estimate_95ci"] = selected.apply(
        lambda row: (
            f"{row['estimate']:.3f} [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
        ),
        axis=1,
    )
    wide = selected.pivot_table(
        index=["lambda", "n"],
        columns="metric",
        values="estimate_95ci",
        aggfunc="first",
    ).reset_index()
    return wide[["lambda", "n", *METRICS]]


def make_figure(metrics: pd.DataFrame, reference_lambda: float, output_path: Path) -> None:
    available = list(metrics["subset"].drop_duplicates())
    strict_candidates = [name for name in available if name.startswith("cross_year_") and name.endswith("unseen_genotype")]
    subsets = ["all_evaluation", strict_candidates[0]] if strict_candidates else ["all_evaluation"]
    fig, axes = plt.subplots(1, len(subsets), figsize=(7.2 * len(subsets), 5.4), squeeze=False)
    colors = {"mae": "#2166ac", "rmse": "#b2182b"}
    labels = {"mae": "MAE", "rmse": "RMSE"}
    for ax, subset_name in zip(axes.flat, subsets, strict=True):
        for metric in ("mae", "rmse"):
            rows = metrics.loc[
                (metrics["subset"] == subset_name) & (metrics["metric"] == metric)
            ].sort_values("lambda")
            ax.plot(
                rows["lambda"],
                rows["estimate"],
                marker="o",
                linewidth=2,
                color=colors[metric],
                label=labels[metric],
            )
            ax.fill_between(
                rows["lambda"],
                rows["ci_lower"],
                rows["ci_upper"],
                color=colors[metric],
                alpha=0.12,
            )
        ax.axvline(reference_lambda, color="#555555", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Dynamic-threshold coefficient lambda")
        ax.set_ylabel("Error (percentage points)")
        ax.set_title(subset_name.replace("_", " ").title())
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    lambdas = sorted(set(float(value) for value in args.lambdas))
    if args.reference_lambda not in lambdas:
        raise ValueError("The reference lambda must be included in --lambdas.")
    if args.n_bootstrap < 100:
        raise ValueError("--n-bootstrap must be at least 100.")

    kernel_df = pd.read_csv(args.kernel_csv)
    plot_df = pd.read_csv(args.plot_csv)
    for frame in (kernel_df, plot_df):
        frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
        frame["plot"] = frame["plot"].astype(str)

    threshold_info = json.loads(args.threshold_info.read_text(encoding="utf-8"))
    healthy_mean = float(threshold_info["mope_h_stats"]["mean"])
    diseased_mean = float(threshold_info["mope_d_stats"]["mean"])
    aggregated = aggregate_predictions(kernel_df, lambdas, healthy_mean, diseased_mean)
    merged = plot_df[["year", "plot", "line", "manual_vsk_pct"]].merge(
        aggregated, on=["year", "plot"], how="left", validate="one_to_one"
    )
    if merged.isna().any().any():
        raise RuntimeError("Threshold sensitivity table contains missing values.")

    metrics, paired = evaluate(
        merged,
        lambdas,
        args.reference_lambda,
        args.training_year,
        args.training_line,
        args.n_bootstrap,
        args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_dir / "per_plot_lambda_predictions.csv", index=False)
    metrics.to_csv(args.output_dir / "lambda_sensitivity_metrics.csv", index=False)
    paired.to_csv(args.output_dir / "paired_reference_lambda_comparisons.csv", index=False)
    build_paper_table(metrics).to_csv(
        args.output_dir / "paper_table_lambda_sensitivity.csv", index=False
    )
    make_figure(metrics, args.reference_lambda, args.output_dir / "lambda_sensitivity.png")

    metadata = {
        "lambdas": lambdas,
        "reference_lambda": args.reference_lambda,
        "healthy_score_mean": healthy_mean,
        "diseased_score_mean": diseased_mean,
        "reference_mean_policy": "Equal mean of the two extreme training classes.",
        "test_labels_used_for_threshold_selection": False,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(build_paper_table(metrics).to_string(index=False))
    print(f"Saved outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
