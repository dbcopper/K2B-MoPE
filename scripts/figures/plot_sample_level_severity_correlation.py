from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "discussion_dataset" / "new" / "run"
RESULTS_ROOT = RUN_ROOT / "results"
PLOTS_ROOT = RUN_ROOT / "plots" / "sample_level_correlation_mu_d"
METADATA_CSV = (
    RUN_ROOT
    / "plots"
    / "dynamic_threshold_analysis"
    / "image_level_dynamic_thresholds.csv"
)

MU_D = 0.8417304754257202
REF_MEAN = 0.3981322646141052
LAMBDA = 0.25


def compute_image_level_table() -> pd.DataFrame:
    meta_df = pd.read_csv(METADATA_CSV)
    rows = []

    for row in meta_df.itertuples(index=False):
        image_stem = Path(row.image_name).stem
        score_csv = RESULTS_ROOT / image_stem / "severity_scores.csv"
        score_df = pd.read_csv(score_csv)
        scores = score_df["severity_score"].astype(float)
        mean_img = float(scores.mean())
        threshold = MU_D + LAMBDA * (mean_img - REF_MEAN)
        predicted_pct = float((scores >= threshold).mean() * 100.0)

        rows.append(
            {
                "group_name": row.group_name,
                "image_name": row.image_name,
                "vsk_value": float(row.vsk_value),
                "Name": row.Name,
                "Pedigree": row.Pedigree,
                "n_kernels": int(len(scores)),
                "mean_img": mean_img,
                "dynamic_threshold": threshold,
                "predicted_severity_pct": predicted_pct,
                "diff_vs_vsk_pct_points": predicted_pct - float(row.vsk_value),
            }
        )

    return pd.DataFrame(rows).sort_values(["group_name", "image_name"]).reset_index(drop=True)


def compute_group_level_table(image_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        image_df.groupby(["group_name", "Name", "Pedigree", "vsk_value"], dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "n_images": int(len(g)),
                    "n_kernels": int(g["n_kernels"].sum()),
                    "mean_img": float((g["mean_img"] * g["n_kernels"]).sum() / g["n_kernels"].sum()),
                    "dynamic_threshold": float(
                        (g["dynamic_threshold"] * g["n_kernels"]).sum() / g["n_kernels"].sum()
                    ),
                    "predicted_severity_pct": float(
                        (g["predicted_severity_pct"] * g["n_kernels"]).sum() / g["n_kernels"].sum()
                    ),
                }
            )
        )
        .reset_index()
    )
    grouped["diff_vs_vsk_pct_points"] = grouped["predicted_severity_pct"] - grouped["vsk_value"]
    return grouped.sort_values("group_name").reset_index(drop=True)


def correlation_stats(df: pd.DataFrame) -> dict[str, float]:
    x = df["vsk_value"].astype(float)
    y = df["predicted_severity_pct"].astype(float)
    rho, rho_p = spearmanr(x, y)
    r, r_p = pearsonr(x, y)
    mae = (y - x).abs().mean()
    return {
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "pearson_r": float(r),
        "pearson_p": float(r_p),
        "mae": float(mae),
    }


def add_panel(ax: plt.Axes, df: pd.DataFrame, title: str, stats: dict[str, float]) -> None:
    x = df["vsk_value"].astype(float)
    y = df["predicted_severity_pct"].astype(float)
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    pad = 2.0

    ax.scatter(
        x,
        y,
        s=70,
        color="#d95f5f",
        edgecolors="white",
        linewidths=0.9,
        alpha=0.82,
    )
    ax.plot(
        [min_val - pad, max_val + pad],
        [min_val - pad, max_val + pad],
        linestyle="--",
        linewidth=2.0,
        color="#6a6a6a",
        label="y = x",
    )

    ax.set_xlim(min_val - pad, max_val + pad)
    ax.set_ylim(min_val - pad, max_val + pad)
    ax.grid(True, alpha=0.22, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(
        f"{title}\nSpearman $\\rho$ = {stats['spearman_rho']:.3f}, Pearson r = {stats['pearson_r']:.3f}",
        fontsize=18,
        pad=10,
    )
    ax.set_xlabel("Manual VSK (%)", fontsize=16)
    ax.set_ylabel("Predicted Severity Score (%)", fontsize=16)
    ax.tick_params(axis="both", labelsize=13, width=1.0, length=5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("#333333")


def make_figure(image_df: pd.DataFrame, group_df: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.titlesize": 18,
            "axes.labelsize": 16,
        }
    )

    image_stats = correlation_stats(image_df)
    group_stats = correlation_stats(group_df)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), constrained_layout=True)
    add_panel(axes[0], image_df, f"(A) Image-level (n = {len(image_df)})", image_stats)
    add_panel(axes[1], group_df, f"(B) Group-level (n = {len(group_df)})", group_stats)
    axes[1].legend(loc="upper left", fontsize=13, frameon=True)

    out_path = PLOTS_ROOT / "sample_level_correlation_dual_panel.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    PLOTS_ROOT.mkdir(parents=True, exist_ok=True)
    image_df = compute_image_level_table()
    group_df = compute_group_level_table(image_df)

    image_df.to_csv(PLOTS_ROOT / "image_level_mu_d_dynamic.csv", index=False)
    group_df.to_csv(PLOTS_ROOT / "group_level_mu_d_dynamic.csv", index=False)

    make_figure(image_df, group_df)

    image_stats = correlation_stats(image_df)
    group_stats = correlation_stats(group_df)
    summary = pd.DataFrame(
        [
            {"level": "image", **image_stats},
            {"level": "group", **group_stats},
        ]
    )
    summary.to_csv(PLOTS_ROOT / "summary_metrics.csv", index=False)

    print("Saved:")
    print(PLOTS_ROOT / "sample_level_correlation_dual_panel.png")
    print(PLOTS_ROOT / "image_level_mu_d_dynamic.csv")
    print(PLOTS_ROOT / "group_level_mu_d_dynamic.csv")
    print(PLOTS_ROOT / "summary_metrics.csv")


if __name__ == "__main__":
    main()
