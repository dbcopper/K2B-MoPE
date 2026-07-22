from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = (
    ROOT / "results" / "result2" / "per_plot" / "table8_per_plot_predictions.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "results" / "result4_generalization"

REQUIRED_COLUMNS = {"year", "plot", "line", "manual_vsk_pct"}
PREDICTION_SUFFIX = "_predicted_vsk_pct"
MODEL_LABELS = {
    "svm_predicted_vsk_pct": "SVM (RBF)",
    "rf_predicted_vsk_pct": "Random Forest",
    "resnet18_predicted_vsk_pct": "ResNet18",
    "vit_predicted_vsk_pct": "ViT",
    "mope_predicted_vsk_pct": "MoPE",
}
METRIC_ORDER = ["pearson_r", "spearman_rho", "mae", "rmse", "bias"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate cross-year and unseen-genotype generalization from an existing "
            "per-plot prediction table. No model retraining or new data are required."
        )
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PREDICTIONS,
        help="CSV containing year, plot, line, manual VSK, and model predictions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for audit tables, statistics, and figures.",
    )
    parser.add_argument(
        "--training-year",
        type=int,
        default=2025,
        help="Year represented by the extreme-label training reference set.",
    )
    parser.add_argument(
        "--training-line",
        default="MN-Rothsay",
        help="Genotype represented by the extreme-label training reference set.",
    )
    parser.add_argument(
        "--reference-model-column",
        default="mope_predicted_vsk_pct",
        help="Prediction column used as the reference in paired comparisons.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=5000,
        help="Number of paired percentile-bootstrap resamples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260703,
        help="Random seed for deterministic bootstrap confidence intervals.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write statistics only.",
    )
    return parser.parse_args()


def canonical_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def model_label(column: str) -> str:
    if column in MODEL_LABELS:
        return MODEL_LABELS[column]
    stem = column.removesuffix(PREDICTION_SUFFIX)
    return stem.replace("_", " ").title()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_predictions(path: Path) -> tuple[pd.DataFrame, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Prediction table not found: {path}")

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    prediction_columns = [
        column for column in df.columns if column.endswith(PREDICTION_SUFFIX)
    ]
    if not prediction_columns:
        raise ValueError(
            f"No prediction columns ending in {PREDICTION_SUFFIX!r} were found."
        )

    clean = df.copy()
    clean["year"] = pd.to_numeric(clean["year"], errors="raise").astype(int)
    clean["plot"] = clean["plot"].astype(str).str.strip()
    clean["line"] = clean["line"].astype(str).str.strip()
    numeric_columns = ["manual_vsk_pct", *prediction_columns]
    for column in numeric_columns:
        clean[column] = pd.to_numeric(clean[column], errors="raise")

    if clean[["year", "plot", "line", *numeric_columns]].isna().any().any():
        raise ValueError("The prediction table contains missing values in required fields.")
    if clean.duplicated(["year", "plot"]).any():
        duplicates = clean.loc[
            clean.duplicated(["year", "plot"], keep=False), ["year", "plot"]
        ]
        raise ValueError(
            "Duplicate year/plot records were found: "
            + duplicates.drop_duplicates().to_dict(orient="records").__repr__()
        )

    return clean.sort_values(["year", "plot"]).reset_index(drop=True), prediction_columns


def build_subsets(
    df: pd.DataFrame,
    training_year: int,
    training_line: str,
) -> dict[str, np.ndarray]:
    line_token = canonical_token(training_line)
    is_training_year = df["year"].to_numpy() == training_year
    is_training_line = (
        df["line"].map(canonical_token).to_numpy(dtype=object) == line_token
    )
    other_years = sorted(int(year) for year in df.loc[~is_training_year, "year"].unique())
    if not other_years:
        raise ValueError(
            f"No cross-year records remain after excluding training year {training_year}."
        )
    year_tag = "_".join(str(year) for year in other_years)

    subsets = {
        "all_evaluation": np.ones(len(df), dtype=bool),
        f"cross_year_{year_tag}": ~is_training_year,
        "unseen_genotype": ~is_training_line,
        f"cross_year_{year_tag}_unseen_genotype": (~is_training_year)
        & (~is_training_line),
        f"same_year_{training_year}_unseen_genotype": is_training_year
        & (~is_training_line),
    }
    for name, mask in subsets.items():
        if int(mask.sum()) < 3:
            raise ValueError(f"Subset {name!r} has fewer than three records.")
    return subsets


def aggregate_line_year(
    df: pd.DataFrame,
    prediction_columns: Iterable[str],
) -> pd.DataFrame:
    value_columns = ["manual_vsk_pct", *prediction_columns]
    aggregated = (
        df.groupby(["year", "line"], as_index=False)[value_columns]
        .mean()
        .sort_values(["year", "line"])
        .reset_index(drop=True)
    )
    counts = (
        df.groupby(["year", "line"], as_index=False)
        .size()
        .rename(columns={"size": "n_plots"})
    )
    return aggregated.merge(counts, on=["year", "line"], how="left")


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = np.sqrt(
        np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered)
    )
    if denominator <= 0.0:
        return float("nan")
    return float(np.dot(x_centered, y_centered) / denominator)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return {
        "pearson_r": safe_pearson(y_true, y_pred),
        "spearman_rho": safe_pearson(rankdata(y_true), rankdata(y_pred)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "bias": float(np.mean(residual)),
    }


def bootstrap_distributions(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    sampled_true = y_true[indices]
    true_ranks = rankdata(sampled_true, axis=1)

    def rowwise_pearson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x_centered = x - x.mean(axis=1, keepdims=True)
        y_centered = y - y.mean(axis=1, keepdims=True)
        numerator = np.sum(x_centered * y_centered, axis=1)
        denominator = np.sqrt(
            np.sum(np.square(x_centered), axis=1)
            * np.sum(np.square(y_centered), axis=1)
        )
        result = np.full(len(x), np.nan, dtype=float)
        np.divide(numerator, denominator, out=result, where=denominator > 0.0)
        return result

    distributions: dict[str, dict[str, np.ndarray]] = {}
    for model, values in predictions.items():
        sampled_prediction = values[indices]
        residual = sampled_prediction - sampled_true
        distributions[model] = {
            "pearson_r": rowwise_pearson(sampled_true, sampled_prediction),
            "spearman_rho": rowwise_pearson(
                true_ranks, rankdata(sampled_prediction, axis=1)
            ),
            "mae": np.mean(np.abs(residual), axis=1),
            "rmse": np.sqrt(np.mean(np.square(residual), axis=1)),
            "bias": np.mean(residual, axis=1),
        }
    return distributions


def percentile_interval(values: np.ndarray) -> tuple[float, float, int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return float("nan"), float("nan"), 0
    lower, upper = np.percentile(finite, [2.5, 97.5])
    return float(lower), float(upper), int(len(finite))


def two_sided_bootstrap_p(improvements: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(improvements, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return float("nan"), float("nan")
    positive_count = int(np.count_nonzero(finite > 0.0))
    nonpositive_count = int(len(finite) - positive_count)
    probability_positive = float(positive_count / len(finite))
    p_value = min(
        1.0,
        2.0 * (min(positive_count, nonpositive_count) + 1) / (len(finite) + 1),
    )
    return probability_positive, p_value


def analyse_level(
    subset_name: str,
    level_name: str,
    level_df: pd.DataFrame,
    prediction_columns: list[str],
    reference_column: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    y_true = level_df["manual_vsk_pct"].to_numpy(dtype=float)
    predictions = {
        column: level_df[column].to_numpy(dtype=float)
        for column in prediction_columns
    }
    sample_indices = rng.integers(
        0, len(level_df), size=(n_bootstrap, len(level_df)), dtype=np.int64
    )
    distributions = bootstrap_distributions(y_true, predictions, sample_indices)

    metric_rows: list[dict[str, object]] = []
    point_metrics: dict[str, dict[str, float]] = {}
    for column, values in predictions.items():
        point_metrics[column] = calculate_metrics(y_true, values)
        for metric in METRIC_ORDER:
            lower, upper, valid_count = percentile_interval(
                distributions[column][metric]
            )
            metric_rows.append(
                {
                    "subset": subset_name,
                    "analysis_level": level_name,
                    "model_column": column,
                    "model": model_label(column),
                    "n": len(level_df),
                    "metric": metric,
                    "estimate": point_metrics[column][metric],
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "valid_bootstrap_resamples": valid_count,
                }
            )

    paired_rows: list[dict[str, object]] = []
    for baseline_column in prediction_columns:
        if baseline_column == reference_column:
            continue
        comparisons = {
            "pearson_r": distributions[reference_column]["pearson_r"]
            - distributions[baseline_column]["pearson_r"],
            "spearman_rho": distributions[reference_column]["spearman_rho"]
            - distributions[baseline_column]["spearman_rho"],
            "mae": distributions[baseline_column]["mae"]
            - distributions[reference_column]["mae"],
            "rmse": distributions[baseline_column]["rmse"]
            - distributions[reference_column]["rmse"],
            "absolute_bias": np.abs(distributions[baseline_column]["bias"])
            - np.abs(distributions[reference_column]["bias"]),
        }
        point_improvements = {
            "pearson_r": point_metrics[reference_column]["pearson_r"]
            - point_metrics[baseline_column]["pearson_r"],
            "spearman_rho": point_metrics[reference_column]["spearman_rho"]
            - point_metrics[baseline_column]["spearman_rho"],
            "mae": point_metrics[baseline_column]["mae"]
            - point_metrics[reference_column]["mae"],
            "rmse": point_metrics[baseline_column]["rmse"]
            - point_metrics[reference_column]["rmse"],
            "absolute_bias": abs(point_metrics[baseline_column]["bias"])
            - abs(point_metrics[reference_column]["bias"]),
        }
        for metric, bootstrap_values in comparisons.items():
            lower, upper, valid_count = percentile_interval(bootstrap_values)
            probability_positive, p_value = two_sided_bootstrap_p(bootstrap_values)
            paired_rows.append(
                {
                    "subset": subset_name,
                    "analysis_level": level_name,
                    "reference_model": model_label(reference_column),
                    "baseline_model": model_label(baseline_column),
                    "n": len(level_df),
                    "metric": metric,
                    "improvement_estimate": point_improvements[metric],
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "probability_improvement_gt_zero": probability_positive,
                    "two_sided_bootstrap_p": p_value,
                    "valid_bootstrap_resamples": valid_count,
                    "interpretation": (
                        "Positive values favour the reference model."
                    ),
                }
            )
    return metric_rows, paired_rows


def build_subset_audit(
    df: pd.DataFrame,
    subsets: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = df[["year", "plot", "line", "manual_vsk_pct"]].copy()
    count_rows = []
    for name, mask in subsets.items():
        manifest[name] = mask
        selected = df.loc[mask]
        count_rows.append(
            {
                "subset": name,
                "n_plots": len(selected),
                "n_lines": selected["line"].nunique(),
                "n_line_year_groups": selected[["year", "line"]]
                .drop_duplicates()
                .shape[0],
                "years": ";".join(str(year) for year in sorted(selected["year"].unique())),
                "lines": ";".join(sorted(selected["line"].unique())),
                "manual_vsk_min": float(selected["manual_vsk_pct"].min()),
                "manual_vsk_max": float(selected["manual_vsk_pct"].max()),
                "manual_vsk_mean": float(selected["manual_vsk_pct"].mean()),
            }
        )
    return manifest, pd.DataFrame(count_rows)


def format_ci(estimate: float, lower: float, upper: float) -> str:
    return f"{estimate:.3f} [{lower:.3f}, {upper:.3f}]"


def build_paper_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    plot_metrics = metrics_df.loc[metrics_df["analysis_level"] == "plot"].copy()
    plot_metrics["estimate_95ci"] = plot_metrics.apply(
        lambda row: format_ci(row["estimate"], row["ci_lower"], row["ci_upper"]),
        axis=1,
    )
    wide = plot_metrics.pivot_table(
        index=["subset", "model", "n"],
        columns="metric",
        values="estimate_95ci",
        aggfunc="first",
    ).reset_index()
    desired_columns = ["subset", "model", "n", *METRIC_ORDER]
    return wide[[column for column in desired_columns if column in wide.columns]]


def lookup_metric(
    metrics_df: pd.DataFrame,
    subset: str,
    model_column: str,
    metric: str,
) -> pd.Series:
    rows = metrics_df.loc[
        (metrics_df["analysis_level"] == "plot")
        & (metrics_df["subset"] == subset)
        & (metrics_df["model_column"] == model_column)
        & (metrics_df["metric"] == metric)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one metric row for {subset}/{model_column}/{metric}; got {len(rows)}."
        )
    return rows.iloc[0]


def make_mope_scatter_figure(
    df: pd.DataFrame,
    subsets: dict[str, np.ndarray],
    metrics_df: pd.DataFrame,
    reference_column: str,
    output_path: Path,
) -> None:
    panel_names = [name for name in subsets if name != "all_evaluation"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.0), constrained_layout=True)
    for ax, subset_name in zip(axes.flat, panel_names, strict=True):
        selected = df.loc[subsets[subset_name]]
        x = selected["manual_vsk_pct"].to_numpy(dtype=float)
        y = selected[reference_column].to_numpy(dtype=float)
        metric = lookup_metric(metrics_df, subset_name, reference_column, "mae")
        pearson = lookup_metric(metrics_df, subset_name, reference_column, "pearson_r")
        lower = min(0.0, float(x.min()), float(y.min()))
        upper = max(100.0, float(x.max()), float(y.max()))
        ax.scatter(
            x,
            y,
            s=54,
            color="#2f6f9f",
            edgecolors="white",
            linewidths=0.7,
            alpha=0.82,
        )
        ax.plot([lower, upper], [lower, upper], "--", color="#666666", linewidth=1.4)
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.18)
        ax.set_title(
            subset_name.replace("_", " ").title()
            + f" (n={len(selected)})\n"
            + f"r={pearson['estimate']:.3f}; "
            + f"MAE={metric['estimate']:.2f} "
            + f"[{metric['ci_lower']:.2f}, {metric['ci_upper']:.2f}]"
        )
        ax.set_xlabel("Manual VSK (%)")
        ax.set_ylabel(f"{model_label(reference_column)} predicted VSK (%)")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_mae_figure(metrics_df: pd.DataFrame, output_path: Path) -> None:
    competitive_models = ["Random Forest", "ResNet18", "ViT", "MoPE"]
    subset_names = [
        name
        for name in metrics_df["subset"].drop_duplicates()
        if name != "all_evaluation"
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5), constrained_layout=True)
    colors = {
        "Random Forest": "#4c78a8",
        "ResNet18": "#f58518",
        "ViT": "#54a24b",
        "MoPE": "#b279a2",
    }
    for ax, subset_name in zip(axes.flat, subset_names, strict=True):
        rows = metrics_df.loc[
            (metrics_df["analysis_level"] == "plot")
            & (metrics_df["subset"] == subset_name)
            & (metrics_df["metric"] == "mae")
            & (metrics_df["model"].isin(competitive_models))
        ].copy()
        rows["model"] = pd.Categorical(
            rows["model"], categories=competitive_models, ordered=True
        )
        rows = rows.sort_values("model")
        positions = np.arange(len(rows))
        lower_error = rows["estimate"].to_numpy() - rows["ci_lower"].to_numpy()
        upper_error = rows["ci_upper"].to_numpy() - rows["estimate"].to_numpy()
        for position, (_, row) in zip(positions, rows.iterrows(), strict=True):
            ax.errorbar(
                row["estimate"],
                position,
                xerr=np.array([[lower_error[position]], [upper_error[position]]]),
                fmt="o",
                markersize=7,
                capsize=4,
                color=colors[str(row["model"])],
            )
        ax.set_yticks(positions, rows["model"].astype(str))
        ax.set_xlabel("MAE (percentage points; 95% bootstrap CI)")
        ax.set_title(subset_name.replace("_", " ").title())
        ax.grid(axis="x", alpha=0.2)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.n_bootstrap < 100:
        raise ValueError("--n-bootstrap must be at least 100.")

    predictions_path = args.predictions.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df, prediction_columns = load_predictions(predictions_path)
    if args.reference_model_column not in prediction_columns:
        raise ValueError(
            f"Reference model column {args.reference_model_column!r} is not present. "
            f"Available prediction columns: {prediction_columns}"
        )

    subsets = build_subsets(df, args.training_year, args.training_line)
    manifest_df, subset_counts_df = build_subset_audit(df, subsets)
    manifest_df.to_csv(output_dir / "subset_manifest.csv", index=False)
    subset_counts_df.to_csv(output_dir / "subset_counts.csv", index=False)

    rng = np.random.default_rng(args.seed)
    all_metric_rows: list[dict[str, object]] = []
    all_paired_rows: list[dict[str, object]] = []
    for subset_name, mask in subsets.items():
        plot_df = df.loc[mask].reset_index(drop=True)
        line_year_df = aggregate_line_year(plot_df, prediction_columns)
        for level_name, level_df in (
            ("plot", plot_df),
            ("line_year_mean", line_year_df),
        ):
            metric_rows, paired_rows = analyse_level(
                subset_name=subset_name,
                level_name=level_name,
                level_df=level_df,
                prediction_columns=prediction_columns,
                reference_column=args.reference_model_column,
                n_bootstrap=args.n_bootstrap,
                rng=rng,
            )
            all_metric_rows.extend(metric_rows)
            all_paired_rows.extend(paired_rows)

    metrics_df = pd.DataFrame(all_metric_rows)
    paired_df = pd.DataFrame(all_paired_rows)
    metrics_df.to_csv(output_dir / "generalization_metrics.csv", index=False)
    paired_df.to_csv(output_dir / "paired_reference_comparisons.csv", index=False)
    build_paper_table(metrics_df).to_csv(
        output_dir / "paper_table_plot_level_95ci.csv", index=False
    )

    if not args.no_plots:
        make_mope_scatter_figure(
            df,
            subsets,
            metrics_df,
            args.reference_model_column,
            output_dir / "mope_generalization_scatter.png",
        )
        make_mae_figure(metrics_df, output_dir / "mae_by_subset.png")

    metadata = {
        "input_csv": str(predictions_path),
        "input_sha256": sha256_file(predictions_path),
        "output_dir": str(output_dir),
        "training_year": args.training_year,
        "training_line": args.training_line,
        "reference_model_column": args.reference_model_column,
        "prediction_columns": prediction_columns,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "confidence_interval": "paired percentile bootstrap, 2.5th to 97.5th percentile",
        "statistical_unit_plot_level": "one multi-kernel image/plot",
        "statistical_unit_line_year_level": "mean of plots within each line-year group",
        "scope": (
            "Cross-year and cross-genotype transfer under the documented, "
            "standardized RGB acquisition protocol."
        ),
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("Subset audit:")
    print(subset_counts_df.to_string(index=False))
    print(f"\nSaved analysis to: {output_dir}")


if __name__ == "__main__":
    main()
