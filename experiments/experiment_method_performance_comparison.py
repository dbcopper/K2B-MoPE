from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, ViT_B_16_Weights, resnet18, vit_b_16


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import few_shot_cellpose_mope_ranking as mope  # noqa: E402
from experiment_generalization_existing_data import (  # noqa: E402
    bootstrap_distributions,
    build_subsets,
    calculate_metrics,
    percentile_interval,
    two_sided_bootstrap_p,
)


DEFAULT_KERNEL_PREDICTIONS = (
    ROOT / "results" / "result2" / "per_kernel" / "table8_per_kernel_predictions.csv"
)
DEFAULT_PLOT_PREDICTIONS = (
    ROOT / "results" / "result2" / "per_plot" / "table8_per_plot_predictions.csv"
)
DEFAULT_ABLATION_PREDICTIONS = (
    ROOT / "results" / "result3" / "per_plot" / "mope_ablation_per_plot_predictions.csv"
)
DEFAULT_TRAINING_ROOT = (
    ROOT / "output_results" / "current_seed_models" / "training_examples"
)
DEFAULT_MODEL_ROOT = ROOT / "results" / "result2" / "models"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "result5_method_comparison"

MODEL_INFO = {
    "svm_standard": {
        "label": "SVM (0.5)",
        "family": "Kernel classifier",
        "threshold": "Fixed 0.5",
        "primary": True,
    },
    "svm_calibrated": {
        "label": "SVM (training-calibrated)",
        "family": "Kernel classifier",
        "threshold": "Grouped OOF training threshold",
        "primary": False,
    },
    "rf_standard": {
        "label": "Random Forest (0.5)",
        "family": "Tree ensemble",
        "threshold": "Fixed 0.5",
        "primary": True,
    },
    "rf_calibrated": {
        "label": "Random Forest (training-calibrated)",
        "family": "Tree ensemble",
        "threshold": "Grouped OOF training threshold",
        "primary": False,
    },
    "resnet18_standard": {
        "label": "ResNet18 (0.5)",
        "family": "CNN",
        "threshold": "Fixed 0.5",
        "primary": True,
    },
    "resnet18_calibrated": {
        "label": "ResNet18 (validation-calibrated)",
        "family": "CNN",
        "threshold": "Held-out kernel validation threshold",
        "primary": False,
    },
    "vit_standard": {
        "label": "ViT (0.5)",
        "family": "Vision Transformer",
        "threshold": "Fixed 0.5",
        "primary": True,
    },
    "vit_calibrated": {
        "label": "ViT (validation-calibrated)",
        "family": "Vision Transformer",
        "threshold": "Held-out kernel validation threshold",
        "primary": False,
    },
    "ranksvm": {
        "label": "RankSVM",
        "family": "Linear pairwise ranking",
        "threshold": "Training-reference dynamic threshold",
        "primary": True,
    },
    "single_linear": {
        "label": "Single-Linear",
        "family": "Neural pairwise ranking",
        "threshold": "Training-reference dynamic threshold",
        "primary": True,
    },
    "mope": {
        "label": "MoPE",
        "family": "Phenotypic mixture of experts",
        "threshold": "Training-reference dynamic threshold",
        "primary": True,
    },
}

METRICS = ["pearson_r", "spearman_rho", "mae", "rmse", "bias"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an expanded, training-only-calibrated method comparison using "
            "cached kernel predictions and the independent 80-plot evaluation set."
        )
    )
    parser.add_argument("--kernel-predictions", type=Path, default=DEFAULT_KERNEL_PREDICTIONS)
    parser.add_argument("--plot-predictions", type=Path, default=DEFAULT_PLOT_PREDICTIONS)
    parser.add_argument("--ablation-predictions", type=Path, default=DEFAULT_ABLATION_PREDICTIONS)
    parser.add_argument("--training-root", type=Path, default=DEFAULT_TRAINING_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--training-year", type=int, default=2025)
    parser.add_argument("--training-line", default="MN-Rothsay")
    parser.add_argument("--rank-svm-c", type=float, default=1.0)
    parser.add_argument("--lambda-value", type=float, default=0.25)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def source_group(path: Path) -> str:
    match = re.match(r"(.+?)_seed_", path.stem, flags=re.IGNORECASE)
    return match.group(1) if match else path.stem


def load_training_features(
    training_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, int]]:
    rows: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    feature_names: list[str] | None = None
    skipped: dict[str, int] = {"Healthy": 0, "Diseased": 0}

    for class_name, label in (("Healthy", 0), ("Diseased", 1)):
        for path in sorted((training_root / class_name).glob("*")):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            image = cv2.imread(str(path))
            if image is None:
                skipped[class_name] += 1
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = binary > 0
            if int(mask.sum()) < 100:
                skipped[class_name] += 1
                continue
            try:
                features = mope.extract_fhb_features(rgb, gray, mask)
                vector, names = mope.get_feature_vector(features)
            except Exception:
                skipped[class_name] += 1
                continue
            rows.append(vector)
            labels.append(label)
            groups.append(source_group(path))
            feature_names = names

    if not rows or feature_names is None:
        raise RuntimeError(f"No valid training features found under {training_root}")
    return (
        np.asarray(rows, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(groups, dtype=object),
        feature_names,
        skipped,
    )


def make_svm(seed: int):
    return make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=seed,
        ),
    )


def make_random_forest(seed: int):
    return RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        min_samples_leaf=2,
    )


def grouped_oof_probabilities(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    svm_probabilities = np.full(len(labels), np.nan, dtype=float)
    rf_probabilities = np.full(len(labels), np.nan, dtype=float)
    splitter = LeaveOneGroupOut()
    for train_indices, validation_indices in splitter.split(features, labels, groups):
        svm = make_svm(seed)
        rf = make_random_forest(seed)
        svm.fit(features[train_indices], labels[train_indices])
        rf.fit(features[train_indices], labels[train_indices])
        svm_probabilities[validation_indices] = svm.predict_proba(
            features[validation_indices]
        )[:, 1]
        rf_probabilities[validation_indices] = rf.predict_proba(
            features[validation_indices]
        )[:, 1]
    if not np.isfinite(svm_probabilities).all() or not np.isfinite(rf_probabilities).all():
        raise RuntimeError("Grouped OOF calibration did not produce all predictions.")
    return svm_probabilities, rf_probabilities


def youden_threshold(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, probabilities)
    finite = np.isfinite(thresholds)
    scores = true_positive_rate[finite] - false_positive_rate[finite]
    candidates = thresholds[finite]
    best_score = float(scores.max())
    tied = candidates[np.isclose(scores, best_score)]
    threshold = float(tied[np.argmin(np.abs(tied - 0.5))])
    predictions = probabilities >= threshold
    healthy_accuracy = float(np.mean(~predictions[labels == 0]))
    diseased_accuracy = float(np.mean(predictions[labels == 1]))
    return {
        "threshold": threshold,
        "balanced_accuracy": (healthy_accuracy + diseased_accuracy) / 2.0,
        "healthy_accuracy": healthy_accuracy,
        "diseased_accuracy": diseased_accuracy,
    }


class ValidationImageDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), torch.tensor(label, dtype=torch.long)


def collect_training_images(training_root: Path) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for class_name, label in (("Healthy", 0), ("Diseased", 1)):
        for path in sorted((training_root / class_name).glob("*")):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                samples.append((path, label))
    return samples


def validation_samples(training_root: Path, seed: int) -> list[tuple[Path, int]]:
    samples = collect_training_images(training_root)
    _train, validation = train_test_split(
        samples,
        test_size=0.2,
        stratify=[label for _path, label in samples],
        random_state=seed,
    )
    return validation


def validation_probabilities(
    model_name: str,
    checkpoint_path: Path,
    samples: list[tuple[Path, int]],
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_name == "vit":
        weights = ViT_B_16_Weights.DEFAULT
        model = vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, 2)
    else:
        raise KeyError(model_name)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    loader = DataLoader(
        ValidationImageDataset(samples, weights.transforms()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for images, batch_labels in loader:
            logits = model(images.to(device))
            probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            labels.append(batch_labels.numpy())
    model.to("cpu")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(labels), np.concatenate(probabilities)


def train_ranksvm(
    features: np.ndarray,
    labels: np.ndarray,
    c_value: float,
    seed: int,
) -> tuple[StandardScaler, LinearSVC, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    normalized = scaler.fit_transform(features)
    healthy = normalized[labels == 0]
    diseased = normalized[labels == 1]
    differences = (
        diseased[:, None, :] - healthy[None, :, :]
    ).reshape(-1, normalized.shape[1]).astype(np.float32)
    pair_features = np.concatenate([differences, -differences], axis=0)
    pair_labels = np.concatenate(
        [np.ones(len(differences), dtype=np.int8), np.zeros(len(differences), dtype=np.int8)]
    )
    classifier = LinearSVC(
        C=c_value,
        class_weight="balanced",
        dual=False,
        max_iter=20000,
        random_state=seed,
    )
    classifier.fit(pair_features, pair_labels)
    training_scores = classifier.decision_function(normalized)
    return (
        scaler,
        classifier,
        training_scores[labels == 0],
        training_scores[labels == 1],
    )


def aggregate_fixed_threshold(
    kernel_df: pd.DataFrame,
    probability_column: str,
    threshold: float,
) -> pd.DataFrame:
    rows = []
    for (year, plot), group in kernel_df.groupby(["year", "plot"], sort=False):
        probabilities = group[probability_column].to_numpy(dtype=float)
        rows.append(
            {
                "year": int(year),
                "plot": str(plot),
                "prediction": float(np.mean(probabilities >= threshold) * 100.0),
            }
        )
    return pd.DataFrame(rows)


def aggregate_rank_scores(
    kernel_df: pd.DataFrame,
    scores: np.ndarray,
    healthy_scores: np.ndarray,
    diseased_scores: np.ndarray,
    lambda_value: float,
) -> pd.DataFrame:
    scored = kernel_df[["year", "plot"]].copy()
    scored["score"] = scores
    reference_mean = float(np.concatenate([healthy_scores, diseased_scores]).mean())
    diseased_mean = float(diseased_scores.mean())
    rows = []
    for (year, plot), group in scored.groupby(["year", "plot"], sort=False):
        sample_scores = group["score"].to_numpy(dtype=float)
        threshold = diseased_mean + lambda_value * (
            float(sample_scores.mean()) - reference_mean
        )
        rows.append(
            {
                "year": int(year),
                "plot": str(plot),
                "prediction": float(np.mean(sample_scores >= threshold) * 100.0),
                "threshold": float(threshold),
                "mean_score": float(sample_scores.mean()),
            }
        )
    return pd.DataFrame(rows)


def add_prediction_column(
    target: pd.DataFrame,
    prediction_df: pd.DataFrame,
    output_column: str,
) -> pd.DataFrame:
    values = prediction_df[["year", "plot", "prediction"]].rename(
        columns={"prediction": output_column}
    )
    merged = target.merge(values, on=["year", "plot"], how="left", validate="one_to_one")
    if merged[output_column].isna().any():
        missing = merged.loc[merged[output_column].isna(), ["year", "plot"]]
        raise RuntimeError(
            f"Missing predictions for {output_column}: {missing.to_dict(orient='records')}"
        )
    return merged


def build_comparison_table(
    kernel_df: pd.DataFrame,
    plot_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    thresholds: dict[str, dict[str, float]],
    rank_prediction_df: pd.DataFrame,
) -> pd.DataFrame:
    comparison = plot_df[["year", "plot", "line", "manual_vsk_pct"]].copy()
    comparison["plot"] = comparison["plot"].astype(str)
    comparison["svm_standard"] = plot_df["svm_predicted_vsk_pct"]
    comparison["rf_standard"] = plot_df["rf_predicted_vsk_pct"]
    comparison["resnet18_standard"] = plot_df["resnet18_predicted_vsk_pct"]
    comparison["vit_standard"] = plot_df["vit_predicted_vsk_pct"]
    comparison["mope"] = plot_df["mope_predicted_vsk_pct"]

    calibrated_sources = {
        "svm_calibrated": ("svm_prob_diseased", thresholds["svm"]["threshold"]),
        "rf_calibrated": ("rf_prob_diseased", thresholds["rf"]["threshold"]),
        "resnet18_calibrated": (
            "resnet18_prob_diseased",
            thresholds["resnet18"]["threshold"],
        ),
        "vit_calibrated": ("vit_prob_diseased", thresholds["vit"]["threshold"]),
    }
    for output_column, (probability_column, threshold) in calibrated_sources.items():
        calibrated = aggregate_fixed_threshold(kernel_df, probability_column, threshold)
        comparison = add_prediction_column(comparison, calibrated, output_column)

    comparison = add_prediction_column(comparison, rank_prediction_df, "ranksvm")
    single_linear = ablation_df[
        ["year", "plot", "Single-Linear_predicted_vsk_pct"]
    ].rename(columns={"Single-Linear_predicted_vsk_pct": "single_linear"})
    comparison = comparison.merge(
        single_linear,
        on=["year", "plot"],
        how="left",
        validate="one_to_one",
    )
    if comparison[list(MODEL_INFO)].isna().any().any():
        raise RuntimeError("The expanded comparison table contains missing predictions.")
    return comparison.sort_values(["year", "plot"]).reset_index(drop=True)


def evaluate_comparison(
    comparison: pd.DataFrame,
    training_year: int,
    training_line: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subsets = build_subsets(comparison, training_year, training_line)
    rng = np.random.default_rng(seed)
    metric_rows = []
    paired_rows = []
    for subset_name, mask in subsets.items():
        selected = comparison.loc[mask].reset_index(drop=True)
        y_true = selected["manual_vsk_pct"].to_numpy(dtype=float)
        predictions = {
            model_key: selected[model_key].to_numpy(dtype=float)
            for model_key in MODEL_INFO
        }
        indices = rng.integers(
            0, len(selected), size=(n_bootstrap, len(selected)), dtype=np.int64
        )
        distributions = bootstrap_distributions(y_true, predictions, indices)
        point_metrics = {
            model_key: calculate_metrics(y_true, values)
            for model_key, values in predictions.items()
        }
        for model_key in MODEL_INFO:
            for metric in METRICS:
                lower, upper, valid_count = percentile_interval(
                    distributions[model_key][metric]
                )
                metric_rows.append(
                    {
                        "subset": subset_name,
                        "model_key": model_key,
                        "model": MODEL_INFO[model_key]["label"],
                        "family": MODEL_INFO[model_key]["family"],
                        "threshold_method": MODEL_INFO[model_key]["threshold"],
                        "primary_comparison": MODEL_INFO[model_key]["primary"],
                        "n": len(selected),
                        "metric": metric,
                        "estimate": point_metrics[model_key][metric],
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "valid_bootstrap_resamples": valid_count,
                    }
                )

        for baseline_key in MODEL_INFO:
            if baseline_key == "mope":
                continue
            improvements = {
                "pearson_r": distributions["mope"]["pearson_r"]
                - distributions[baseline_key]["pearson_r"],
                "spearman_rho": distributions["mope"]["spearman_rho"]
                - distributions[baseline_key]["spearman_rho"],
                "mae": distributions[baseline_key]["mae"]
                - distributions["mope"]["mae"],
                "rmse": distributions[baseline_key]["rmse"]
                - distributions["mope"]["rmse"],
                "absolute_bias": np.abs(distributions[baseline_key]["bias"])
                - np.abs(distributions["mope"]["bias"]),
            }
            point_improvements = {
                "pearson_r": point_metrics["mope"]["pearson_r"]
                - point_metrics[baseline_key]["pearson_r"],
                "spearman_rho": point_metrics["mope"]["spearman_rho"]
                - point_metrics[baseline_key]["spearman_rho"],
                "mae": point_metrics[baseline_key]["mae"]
                - point_metrics["mope"]["mae"],
                "rmse": point_metrics[baseline_key]["rmse"]
                - point_metrics["mope"]["rmse"],
                "absolute_bias": abs(point_metrics[baseline_key]["bias"])
                - abs(point_metrics["mope"]["bias"]),
            }
            for metric, values in improvements.items():
                lower, upper, valid_count = percentile_interval(values)
                probability, p_value = two_sided_bootstrap_p(values)
                paired_rows.append(
                    {
                        "subset": subset_name,
                        "reference_model": "MoPE",
                        "baseline_model": MODEL_INFO[baseline_key]["label"],
                        "baseline_key": baseline_key,
                        "n": len(selected),
                        "metric": metric,
                        "improvement_estimate": point_improvements[metric],
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "probability_improvement_gt_zero": probability,
                        "two_sided_bootstrap_p": p_value,
                        "valid_bootstrap_resamples": valid_count,
                        "interpretation": "Positive values favour MoPE.",
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(paired_rows)


def format_ci(estimate: float, lower: float, upper: float) -> str:
    return f"{estimate:.3f} [{lower:.3f}, {upper:.3f}]"


def build_paper_table(metrics_df: pd.DataFrame, primary_only: bool) -> pd.DataFrame:
    selected = metrics_df.loc[metrics_df["subset"] == "all_evaluation"].copy()
    if primary_only:
        selected = selected.loc[selected["primary_comparison"]]
    selected["estimate_95ci"] = selected.apply(
        lambda row: format_ci(row["estimate"], row["ci_lower"], row["ci_upper"]),
        axis=1,
    )
    index_columns = ["model", "family", "threshold_method", "n"]
    wide = selected.pivot_table(
        index=index_columns,
        columns="metric",
        values="estimate_95ci",
        aggfunc="first",
    ).reset_index()
    return wide[index_columns + METRICS]


def make_primary_mae_figure(metrics_df: pd.DataFrame, output_path: Path) -> None:
    rows = metrics_df.loc[
        (metrics_df["subset"] == "all_evaluation")
        & (metrics_df["metric"] == "mae")
        & metrics_df["primary_comparison"]
        & (metrics_df["model_key"] != "svm_standard")
    ].copy()
    rows = rows.sort_values("estimate", ascending=True).reset_index(drop=True)
    positions = np.arange(len(rows))
    lower = rows["estimate"].to_numpy() - rows["ci_lower"].to_numpy()
    upper = rows["ci_upper"].to_numpy() - rows["estimate"].to_numpy()
    colors = ["#b2182b" if model == "MoPE" else "#4c78a8" for model in rows["model"]]
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    for position, (_, row) in zip(positions, rows.iterrows(), strict=True):
        ax.errorbar(
            row["estimate"],
            position,
            xerr=np.array([[lower[position]], [upper[position]]]),
            fmt="o",
            markersize=8,
            capsize=4,
            color=colors[position],
        )
    ax.set_yticks(positions)
    ax.set_yticklabels(rows["model"].tolist())
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlabel("MAE (percentage points; 95% bootstrap CI)")
    ax.set_title("Competitive sample-level VSK method comparison")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.n_bootstrap < 100:
        raise ValueError("--n-bootstrap must be at least 100.")
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    kernel_df = pd.read_csv(args.kernel_predictions)
    plot_df = pd.read_csv(args.plot_predictions)
    ablation_df = pd.read_csv(args.ablation_predictions)
    for frame in (kernel_df, plot_df, ablation_df):
        frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
        frame["plot"] = frame["plot"].astype(str)

    training_features, training_labels, training_groups, feature_names, skipped = (
        load_training_features(args.training_root)
    )
    svm_oof, rf_oof = grouped_oof_probabilities(
        training_features, training_labels, training_groups, args.seed
    )
    thresholds = {
        "svm": youden_threshold(training_labels, svm_oof),
        "rf": youden_threshold(training_labels, rf_oof),
    }

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    validation = validation_samples(args.training_root, args.seed)
    resnet_labels, resnet_probabilities = validation_probabilities(
        "resnet18",
        args.model_root / "resnet18" / "resnet18_seed_binary.pth",
        validation,
        args.batch_size,
        device,
    )
    vit_labels, vit_probabilities = validation_probabilities(
        "vit",
        args.model_root / "vit_b16_seed_binary.pth",
        validation,
        args.batch_size,
        device,
    )
    thresholds["resnet18"] = youden_threshold(resnet_labels, resnet_probabilities)
    thresholds["vit"] = youden_threshold(vit_labels, vit_probabilities)

    scaler, rank_model, healthy_scores, diseased_scores = train_ranksvm(
        training_features, training_labels, args.rank_svm_c, args.seed
    )
    test_features = kernel_df[feature_names].to_numpy(dtype=np.float64)
    rank_scores = rank_model.decision_function(scaler.transform(test_features))
    rank_prediction_df = aggregate_rank_scores(
        kernel_df,
        rank_scores,
        healthy_scores,
        diseased_scores,
        args.lambda_value,
    )

    comparison = build_comparison_table(
        kernel_df, plot_df, ablation_df, thresholds, rank_prediction_df
    )
    metrics_df, paired_df = evaluate_comparison(
        comparison,
        args.training_year,
        args.training_line,
        args.n_bootstrap,
        args.seed,
    )

    comparison.to_csv(args.output_dir / "per_plot_predictions_extended.csv", index=False)
    metrics_df.to_csv(args.output_dir / "method_comparison_metrics.csv", index=False)
    paired_df.to_csv(args.output_dir / "paired_mope_comparisons.csv", index=False)
    build_paper_table(metrics_df, primary_only=True).to_csv(
        args.output_dir / "paper_table_primary_methods.csv", index=False
    )
    build_paper_table(metrics_df, primary_only=False).to_csv(
        args.output_dir / "paper_table_all_methods.csv", index=False
    )
    make_primary_mae_figure(
        metrics_df, args.output_dir / "primary_method_mae_95ci.png"
    )

    calibration_summary = {
        "calibration_policy": (
            "No manual VSK values from the 80-plot evaluation set were used to "
            "select any threshold."
        ),
        "classical_models": (
            "Threshold selected by Youden index from leave-one-parent-image-out "
            "OOF predictions on the extreme-label training reference."
        ),
        "deep_models": (
            "Threshold selected by Youden index on the held-out validation split "
            "used by the existing ResNet18 and ViT training scripts."
        ),
        "training_feature_count": int(len(training_features)),
        "training_class_counts": {
            "healthy": int(np.sum(training_labels == 0)),
            "diseased": int(np.sum(training_labels == 1)),
        },
        "training_parent_image_count": int(len(np.unique(training_groups))),
        "skipped_feature_images": skipped,
        "deep_validation_count": int(len(validation)),
        "thresholds": thresholds,
        "ranksvm": {
            "C": args.rank_svm_c,
            "pair_count_one_direction": int(
                np.sum(training_labels == 0) * np.sum(training_labels == 1)
            ),
            "healthy_score_mean": float(healthy_scores.mean()),
            "diseased_score_mean": float(diseased_scores.mean()),
            "lambda": args.lambda_value,
        },
        "bootstrap": {"n": args.n_bootstrap, "seed": args.seed},
        "device": str(device),
    }
    (args.output_dir / "calibration_and_run_metadata.json").write_text(
        json.dumps(calibration_summary, indent=2), encoding="utf-8"
    )

    overall = build_paper_table(metrics_df, primary_only=True)
    print("Training-only thresholds:")
    print(json.dumps(thresholds, indent=2))
    print("\nPrimary comparison:")
    print(overall.to_string(index=False))
    print(f"\nSaved outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
