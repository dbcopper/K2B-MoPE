from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "result8_runtime_uncertainty"
METHOD_FILES = {
    "SeedPose + Cellpose cyto": ROOT / "results" / "result1" / "original_size1" / "per_image" / "seedpose_cyto_metrics.csv",
    "Cellpose cyto": ROOT / "results" / "result1" / "original_size" / "per_image" / "cellpose_cyto_metrics.csv",
    "Cellpose cyto2": ROOT / "results" / "result1" / "original_size" / "per_image" / "cellpose_cyto2_metrics.csv",
    "SAM AMG": ROOT / "results" / "result1" / "original_size_sam_cache" / "per_image" / "sam_amg_metrics.csv",
    "FastSAM AMG": ROOT / "results" / "result1" / "original_size_fastsam_cache" / "per_image" / "fastsam_amg_metrics.csv",
}
REFERENCE_METHOD = "SeedPose + Cellpose cyto"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap runtime and GPU-memory uncertainty across 80 images."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260703)
    return parser.parse_args()


def canonicalize_plot_ids(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["year"] = pd.to_numeric(result["year"], errors="raise").astype(int)
    result["plot"] = result["plot"].astype(str).str.strip()
    return result


def load_method(label: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    raw = pd.read_csv(path)
    raw = canonicalize_plot_ids(raw)
    for column in ("runtime_s", "peak_gpu_mem_mb"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    raw["failure_flag"] = raw["failure_flag"].astype(str).str.casefold().eq("true")
    raw["method_label"] = label

    duplicates = raw.duplicated(["year", "plot"], keep=False)
    if duplicates.any():
        raise ValueError(f"Duplicate year/plot rows in {path}")
    if raw.duplicated(["year", "plot"]).any():
        raise ValueError(f"Unresolved duplicate year/plot rows in {path}")
    return raw


def interval(values: np.ndarray) -> tuple[float, float]:
    valid = values[np.isfinite(values)]
    return tuple(np.percentile(valid, [2.5, 97.5])) if len(valid) else (np.nan, np.nan)


def bootstrap_summary(
    values: np.ndarray, indices: np.ndarray
) -> tuple[float, float, float, float, float]:
    means = values[indices].mean(axis=1)
    lower, upper = interval(means)
    return float(values.mean()), float(np.median(values)), float(values.std(ddof=1)), lower, upper


def make_figure(summary: pd.DataFrame, paired: pd.DataFrame, output_path: Path) -> None:
    overall = summary.loc[summary["subset"] == "all_80_images"].copy()
    overall = overall.sort_values("mean_runtime_s")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    y = np.arange(len(overall))
    axes[0].errorbar(
        overall["mean_runtime_s"],
        y,
        xerr=np.vstack(
            [
                overall["mean_runtime_s"] - overall["runtime_ci_lower"],
                overall["runtime_ci_upper"] - overall["mean_runtime_s"],
            ]
        ),
        fmt="o",
        capsize=3,
        color="#2166ac",
    )
    axes[0].set_yticks(y, overall["method"])
    axes[0].set_xlabel("Runtime per image (s; 95% bootstrap CI)")
    axes[0].grid(axis="x", alpha=0.25)

    paired = paired.sort_values("mean_speedup_vs_seedpose")
    y2 = np.arange(len(paired))
    axes[1].errorbar(
        paired["mean_speedup_vs_seedpose"],
        y2,
        xerr=np.vstack(
            [
                paired["mean_speedup_vs_seedpose"] - paired["speedup_ci_lower"],
                paired["speedup_ci_upper"] - paired["mean_speedup_vs_seedpose"],
            ]
        ),
        fmt="o",
        capsize=3,
        color="#b2182b",
    )
    axes[1].axvline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_yticks(y2, paired["comparison_method"])
    axes[1].set_xlabel("Paired runtime ratio (comparison / SeedPose)")
    axes[1].grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.n_bootstrap < 100:
        raise ValueError("--n-bootstrap must be at least 100")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = [load_method(label, path) for label, path in METHOD_FILES.items()]
    reference_keys = set(
        map(tuple, frames[0][["year", "plot"]].itertuples(index=False, name=None))
    )
    if len(reference_keys) != 80:
        raise ValueError(f"Expected 80 reference images, found {len(reference_keys)}")
    for frame in frames[1:]:
        keys = set(map(tuple, frame[["year", "plot"]].itertuples(index=False, name=None)))
        if keys != reference_keys:
            raise ValueError(
                f"Image set mismatch for {frame['method_label'].iloc[0]}: "
                f"missing={sorted(reference_keys - keys)}, extra={sorted(keys - reference_keys)}"
            )
    all_data = pd.concat(frames, ignore_index=True)
    if all_data[["runtime_s", "peak_gpu_mem_mb"]].isna().any().any():
        raise ValueError("Missing runtime or memory measurements were found")

    rng = np.random.default_rng(args.seed)
    subsets = {
        "all_80_images": lambda frame: np.ones(len(frame), dtype=bool),
        "year_2024": lambda frame: frame["year"].to_numpy() == 2024,
        "year_2025": lambda frame: frame["year"].to_numpy() == 2025,
    }
    summary_rows = []
    for method, group in all_data.groupby("method_label", sort=False):
        for subset_name, selector in subsets.items():
            selected = group.loc[selector(group)].copy()
            n = len(selected)
            indices = rng.integers(0, n, size=(args.n_bootstrap, n), dtype=np.int64)
            runtime = selected["runtime_s"].to_numpy(dtype=float)
            memory = selected["peak_gpu_mem_mb"].to_numpy(dtype=float)
            mean_rt, median_rt, sd_rt, rt_lo, rt_hi = bootstrap_summary(runtime, indices)
            mean_mem, median_mem, sd_mem, mem_lo, mem_hi = bootstrap_summary(memory, indices)
            summary_rows.append(
                {
                    "method": method,
                    "subset": subset_name,
                    "n_images": n,
                    "mean_runtime_s": mean_rt,
                    "median_runtime_s": median_rt,
                    "runtime_sd_s": sd_rt,
                    "runtime_ci_lower": rt_lo,
                    "runtime_ci_upper": rt_hi,
                    "mean_peak_gpu_mem_mb": mean_mem,
                    "median_peak_gpu_mem_mb": median_mem,
                    "gpu_mem_sd_mb": sd_mem,
                    "gpu_mem_ci_lower": mem_lo,
                    "gpu_mem_ci_upper": mem_hi,
                    "failures": int(selected["failure_flag"].sum()),
                }
            )
    summary = pd.DataFrame(summary_rows)

    wide = all_data.pivot(index=["year", "plot"], columns="method_label", values="runtime_s")
    paired_rows = []
    reference = wide[REFERENCE_METHOD].to_numpy(dtype=float)
    indices = rng.integers(0, len(wide), size=(args.n_bootstrap, len(wide)), dtype=np.int64)
    for method in METHOD_FILES:
        if method == REFERENCE_METHOD:
            continue
        comparison = wide[method].to_numpy(dtype=float)
        differences = comparison - reference
        ratios = comparison / reference
        boot_difference = differences[indices].mean(axis=1)
        boot_ratio = ratios[indices].mean(axis=1)
        diff_lo, diff_hi = interval(boot_difference)
        ratio_lo, ratio_hi = interval(boot_ratio)
        paired_rows.append(
            {
                "reference_method": REFERENCE_METHOD,
                "comparison_method": method,
                "n_paired_images": len(wide),
                "mean_runtime_difference_s": float(differences.mean()),
                "difference_ci_lower": diff_lo,
                "difference_ci_upper": diff_hi,
                "mean_speedup_vs_seedpose": float(ratios.mean()),
                "speedup_ci_lower": ratio_lo,
                "speedup_ci_upper": ratio_hi,
                "fraction_seedpose_faster": float(np.mean(reference < comparison)),
            }
        )
    paired = pd.DataFrame(paired_rows)

    all_data.to_csv(args.output_dir / "aligned_per_image_runtime.csv", index=False)
    summary.to_csv(args.output_dir / "runtime_memory_summary.csv", index=False)
    paired.to_csv(args.output_dir / "paired_runtime_comparisons.csv", index=False)
    paper = summary.loc[summary["subset"] == "all_80_images"].copy()
    paper["runtime_mean_95ci_s"] = paper.apply(
        lambda row: f"{row['mean_runtime_s']:.2f} [{row['runtime_ci_lower']:.2f}, {row['runtime_ci_upper']:.2f}]",
        axis=1,
    )
    paper["gpu_memory_mean_95ci_mb"] = paper.apply(
        lambda row: f"{row['mean_peak_gpu_mem_mb']:.1f} [{row['gpu_mem_ci_lower']:.1f}, {row['gpu_mem_ci_upper']:.1f}]",
        axis=1,
    )
    paper[["method", "n_images", "runtime_mean_95ci_s", "gpu_memory_mean_95ci_mb", "failures"]].to_csv(
        args.output_dir / "paper_runtime_table.csv", index=False
    )
    make_figure(summary, paired, args.output_dir / "runtime_uncertainty.png")
    metadata = {
        "n_bootstrap": args.n_bootstrap,
        "bootstrap_seed": args.seed,
        "unit_of_resampling": "image",
        "design": "Paired across the same 80 full-resolution images",
        "plot_id_correction": "All retained runtime records use corrected plot identifier 15131.",
        "limitation": "One recorded timing per image estimates across-image uncertainty, not repeated-run timing jitter.",
        "source_files": {label: str(path) for label, path in METHOD_FILES.items()},
    }
    (args.output_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Wrote runtime uncertainty outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
