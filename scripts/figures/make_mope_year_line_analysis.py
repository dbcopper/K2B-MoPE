from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results" / "result2" / "per_plot" / "table8_per_plot_predictions.csv"
OUT = ROOT / "discussions" / "discussion1"


def metric_row(group_name: str, df: pd.DataFrame) -> dict:
    y = df["manual_vsk_pct"].to_numpy(dtype=float)
    p = df["mope_predicted_vsk_pct"].to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    y = y[valid]
    p = p[valid]
    if len(y) >= 2 and len(np.unique(y)) > 1 and len(np.unique(p)) > 1:
        pr, pp = pearsonr(y, p)
        sr, sp = spearmanr(y, p)
    else:
        pr, pp, sr, sp = np.nan, np.nan, np.nan, np.nan
    return {
        "group": group_name,
        "n": int(len(y)),
        "manual_vsk_mean": float(np.mean(y)),
        "predicted_vsk_mean": float(np.mean(p)),
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_rho": float(sr),
        "spearman_p": float(sp),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(math.sqrt(mean_squared_error(y, p))),
        "bias": float(np.mean(p - y)),
    }


def save_year_line_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = OUT / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    year_rows = [metric_row(str(year), g) for year, g in df.groupby("year")]
    year_rows.append(metric_row("Overall", df))
    year_df = pd.DataFrame(year_rows)
    year_df.to_csv(tables / "mope_year_level_performance.csv", index=False)

    line_rows = []
    for line, g in df.groupby("line", dropna=False):
        line_name = "" if pd.isna(line) else str(line)
        y = g["manual_vsk_pct"].to_numpy(dtype=float)
        p = g["mope_predicted_vsk_pct"].to_numpy(dtype=float)
        line_rows.append(
            {
                "line": line_name,
                "n": int(len(g)),
                "manual_vsk_mean": float(np.mean(y)),
                "predicted_vsk_mean": float(np.mean(p)),
                "mae": float(mean_absolute_error(y, p)),
                "rmse": float(math.sqrt(mean_squared_error(y, p))),
                "bias": float(np.mean(p - y)),
                "manual_vsk_min": float(np.min(y)),
                "manual_vsk_max": float(np.max(y)),
            }
        )
    line_df = pd.DataFrame(line_rows).sort_values(["n", "mae"], ascending=[False, True])
    line_df.to_csv(tables / "mope_line_level_error_summary.csv", index=False)
    return year_df, line_df


def save_figures(df: pd.DataFrame, year_df: pd.DataFrame) -> None:
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    colors = {"2024": "#1f77b4", "2025": "#d62728"}
    markers = {"2024": "o", "2025": "s"}
    y = df["manual_vsk_pct"].to_numpy(dtype=float)
    p = df["mope_predicted_vsk_pct"].to_numpy(dtype=float)
    residual = p - y
    pr, _ = pearsonr(y, p)
    sr, _ = spearmanr(y, p)
    mae = mean_absolute_error(y, p)
    rmse = math.sqrt(mean_squared_error(y, p))
    bias = float(np.mean(residual))

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.4))
    for year, g in df.groupby("year"):
        year = str(year)
        axes[0].scatter(
            g["manual_vsk_pct"],
            g["mope_predicted_vsk_pct"],
            s=54,
            alpha=0.82,
            color=colors.get(year, "#777777"),
            marker=markers.get(year, "o"),
            edgecolor="white",
            linewidth=0.6,
            label=f"{year} (n={len(g)})",
        )
        axes[1].scatter(
            g["manual_vsk_pct"],
            g["mope_predicted_vsk_pct"] - g["manual_vsk_pct"],
            s=54,
            alpha=0.82,
            color=colors.get(year, "#777777"),
            marker=markers.get(year, "o"),
            edgecolor="white",
            linewidth=0.6,
            label=f"{year} (n={len(g)})",
        )

    axes[0].plot([0, 100], [0, 100], "--", color="#555555", linewidth=1.2)
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(0, 100)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("Manual VSK (%)")
    axes[0].set_ylabel("MoPE-predicted VSK (%)")
    axes[0].set_title("A. Sample-level VSK estimation", loc="left", fontweight="bold")
    axes[0].text(
        0.04,
        0.96,
        f"r = {pr:.3f}\nrho = {sr:.3f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#d0d0d0", alpha=0.9),
    )

    axes[1].axhline(0, linestyle="--", color="#555555", linewidth=1.2)
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Manual VSK (%)")
    axes[1].set_ylabel("Prediction error (Predicted - Manual)")
    axes[1].set_title("B. Prediction residuals", loc="left", fontweight="bold")
    axes[1].text(
        0.04,
        0.96,
        f"Bias = {bias:.2f}",
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#d0d0d0", alpha=0.9),
    )

    for ax in axes:
        ax.grid(alpha=0.18)
        ax.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=10)

    fig.tight_layout()
    fig.savefig(fig_dir / "mope_vsk_by_year_scatter_residual.png", dpi=300)
    fig.savefig(fig_dir / "mope_vsk_by_year_scatter_residual.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    year_plot = year_df[year_df["group"] != "Overall"].copy()
    x = np.arange(len(year_plot))
    ax.bar(x - 0.18, year_plot["mae"], width=0.36, color="#9ecae1", label="MAE")
    ax.bar(x + 0.18, year_plot["rmse"], width=0.36, color="#3182bd", label="RMSE")
    ax.axhline(float(year_df.loc[year_df["group"] == "Overall", "mae"].iloc[0]), color="#9ecae1", linestyle="--", linewidth=1.2, label="Overall MAE")
    ax.set_xticks(x)
    ax.set_xticklabels(year_plot["group"])
    ax.set_ylabel("Error (percentage points)")
    ax.set_title("MoPE Error by Year", fontweight="bold")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "mope_year_level_error_bars.png", dpi=300)
    fig.savefig(fig_dir / "mope_year_level_error_bars.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT)
    df["year"] = df["year"].astype(str)
    df["plot"] = df["plot"].astype(str)
    year_df, line_df = save_year_line_tables(df)
    save_figures(df, year_df)
    summary = {
        "input": str(INPUT),
        "n_plots": int(len(df)),
        "years": {str(k): int(v) for k, v in df.groupby("year").size().items()},
        "outputs": {
            "year_table": str(OUT / "tables" / "mope_year_level_performance.csv"),
            "line_table": str(OUT / "tables" / "mope_line_level_error_summary.csv"),
            "scatter_residual": str(OUT / "figures" / "mope_vsk_by_year_scatter_residual.png"),
        },
    }
    (OUT / "discussion1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(year_df.to_string(index=False))
    print(f"Saved outputs to {OUT}")


if __name__ == "__main__":
    main()
