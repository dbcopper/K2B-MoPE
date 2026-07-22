from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "discussions" / "discussion1"
INPUT = ROOT / "results" / "result2" / "per_plot" / "table8_per_plot_predictions.csv"

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False


def rmse(values: pd.Series) -> float:
    arr = values.to_numpy(dtype=float)
    return float(np.sqrt(np.mean(arr**2)))


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["error"] = df["mope_predicted_vsk_pct"] - df["manual_vsk_pct"]
    summary = (
        df.groupby(["line", "year"], as_index=False)
        .agg(
            n_plots=("plot", "count"),
            manual_vsk_mean=("manual_vsk_pct", "mean"),
            mope_vsk_mean=("mope_predicted_vsk_pct", "mean"),
            mae=("error", lambda x: float(np.mean(np.abs(x)))),
            rmse=("error", rmse),
            bias=("error", "mean"),
            manual_vsk_min=("manual_vsk_pct", "min"),
            manual_vsk_max=("manual_vsk_pct", "max"),
        )
        .sort_values(["line", "year"])
    )
    return summary


def plot_line_year(summary: pd.DataFrame) -> None:
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    line_order = (
        summary.groupby("line")["manual_vsk_mean"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    years = sorted(summary["year"].astype(str).unique())
    colors = {"2024": "#3B82B6", "2025": "#E78632"}
    x = np.arange(len(line_order))
    width = 0.36 if len(years) == 2 else 0.72 / max(1, len(years))

    fig, ax = plt.subplots(figsize=(12.5, 5.8))

    for yi, year in enumerate(years):
        offset = (yi - (len(years) - 1) / 2) * width
        part = summary[summary["year"].astype(str) == year].set_index("line")
        xs = x + offset
        manual_vals = []
        pred_vals = []
        present_xs = []

        for j, line in enumerate(line_order):
            if line in part.index:
                row = part.loc[line]
                manual = float(row["manual_vsk_mean"])
                pred = float(row["mope_vsk_mean"])
                manual_vals.append(manual)
                pred_vals.append(pred)
                present_xs.append(xs[j])
            else:
                manual_vals.append(np.nan)
                pred_vals.append(np.nan)

        present_xs = np.asarray(present_xs, dtype=float)
        manual_clean = np.asarray([v for v in manual_vals if not np.isnan(v)], dtype=float)
        pred_clean = np.asarray([v for v in pred_vals if not np.isnan(v)], dtype=float)

        ax.bar(
            present_xs,
            manual_clean,
            width=width * 0.86,
            color=colors.get(year, "#777777"),
            alpha=0.28,
            edgecolor=colors.get(year, "#777777"),
            linewidth=1.5,
            label=f"{year} manual VSK",
        )
        ax.scatter(
            present_xs,
            pred_clean,
            s=72,
            color=colors.get(year, "#777777"),
            edgecolor="black",
            linewidth=0.7,
            marker="D" if year == "2024" else "o",
            zorder=4,
            label=f"{year} MoPE predicted",
        )
        for px, manual, pred in zip(present_xs, manual_clean, pred_clean):
            ax.plot(
                [px, px],
                [manual, pred],
                color=colors.get(year, "#777777"),
                alpha=0.8,
                linewidth=1.6,
                zorder=3,
            )

    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(line_order, rotation=35, ha="right", fontsize=11)
    ax.set_ylabel("VSK (%)", fontsize=13)
    ax.set_xlabel("Line", fontsize=13)
    ax.set_title("Line- and year-level manual VSK vs MoPE-predicted VSK", fontsize=15, pad=12)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", frameon=True, fontsize=10, ncol=2)

    fig.tight_layout()
    fig.savefig(fig_dir / "mope_line_year_manual_vs_predicted.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "mope_line_year_manual_vs_predicted.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_line_year_i_shape(summary: pd.DataFrame) -> None:
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    line_order = (
        summary.groupby("line")["manual_vsk_mean"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    years = sorted(summary["year"].astype(str).unique())
    colors = {"2024": "#2563A6", "2025": "#D97706"}
    x = np.arange(len(line_order))
    width = 0.34 if len(years) == 2 else 0.72 / max(1, len(years))
    cap = width * 0.28

    fig, ax = plt.subplots(figsize=(12.5, 5.8))

    for yi, year in enumerate(years):
        offset = (yi - (len(years) - 1) / 2) * width
        part = summary[summary["year"].astype(str) == year].set_index("line")
        color = colors.get(year, "#777777")

        for j, line in enumerate(line_order):
            if line not in part.index:
                continue
            row = part.loc[line]
            xpos = x[j] + offset
            manual = float(row["manual_vsk_mean"])
            pred = float(row["mope_vsk_mean"])

            ax.plot([xpos, xpos], [manual, pred], color=color, linewidth=2.1, alpha=0.82, zorder=2)
            ax.plot([xpos - cap, xpos + cap], [manual, manual], color="black", linewidth=3.0, zorder=4)
            ax.plot([xpos - cap, xpos + cap], [pred, pred], color=color, linewidth=3.0, zorder=4)
            ax.scatter(
                [xpos],
                [pred],
                s=42,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                zorder=5,
            )

    # Compact custom legend.
    ax.plot([], [], color="black", linewidth=3.0, label="Manual VSK mean")
    ax.plot([], [], color="#666666", linewidth=2.1, label="Prediction error")
    ax.plot([], [], color=colors.get("2024", "#777777"), linewidth=3.0, label="MoPE 2024")
    ax.plot([], [], color=colors.get("2025", "#777777"), linewidth=3.0, label="MoPE 2025")

    ax.set_xticks(x)
    ax.set_xticklabels(line_order, rotation=35, ha="right", fontsize=11)
    ax.set_ylabel("VSK (%)", fontsize=13)
    ax.set_xlabel("Line", fontsize=13)
    ax.set_title("Line-year residual structure between manual VSK and MoPE prediction", fontsize=15, pad=12)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", frameon=True, fontsize=10, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "mope_line_year_i_shape_residual.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "mope_line_year_i_shape_residual.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_line_year_dumbbell(summary: pd.DataFrame) -> None:
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    years = sorted(summary["year"].astype(str).unique())
    colors = {"2024": "#2563A6", "2025": "#D97706"}
    fig, axes = plt.subplots(1, len(years), figsize=(12.8, 6.2), sharex=True)
    if len(years) == 1:
        axes = [axes]

    x_max = max(summary["manual_vsk_mean"].max(), summary["mope_vsk_mean"].max()) + 8
    x_max = max(55, min(85, x_max))

    for ax, year in zip(axes, years):
        part = summary[summary["year"].astype(str) == year].copy()
        part = part.sort_values("manual_vsk_mean", ascending=True).reset_index(drop=True)
        y = np.arange(len(part))
        color = colors.get(year, "#777777")

        for i, row in part.iterrows():
            manual = float(row["manual_vsk_mean"])
            pred = float(row["mope_vsk_mean"])
            ax.plot([manual, pred], [i, i], color="#B8B8B8", linewidth=2.0, zorder=1)
            ax.scatter(
                manual,
                i,
                s=72,
                facecolor="white",
                edgecolor="#2F2F2F",
                linewidth=1.6,
                zorder=3,
            )
            ax.scatter(
                pred,
                i,
                s=72,
                facecolor=color,
                edgecolor="white",
                linewidth=0.9,
                zorder=4,
            )

        ax.set_yticks(y)
        ax.set_yticklabels(part["line"], fontsize=14)
        ax.set_title(year, fontsize=18, fontweight="bold")
        ax.set_xlabel("VSK (%)", fontsize=16)
        ax.set_xlim(0, x_max)
        ax.grid(axis="x", color="#E5E5E5", linewidth=0.9)
        ax.grid(axis="y", visible=False)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#444444")

        line_year_errors = part["mope_vsk_mean"] - part["manual_vsk_mean"]
        mae = float(np.mean(np.abs(line_year_errors)))
        rmse_value = float(np.sqrt(np.mean(line_year_errors**2)))
        ax.tick_params(axis="x", labelsize=13)
        ax.text(
            0.98,
            0.06,
            f"line-level MAE = {mae:.1f}\nRMSE = {rmse_value:.1f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=13,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#DDDDDD", alpha=0.9),
        )

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#2F2F2F", markeredgewidth=1.6, markersize=8, label="Manual VSK"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#666666", markeredgecolor="white", markersize=8, label="MoPE predicted VSK"),
        plt.Line2D([0], [0], color="#B8B8B8", linewidth=2.0, label="Prediction residual"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=14, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Line-year comparison between manual and MoPE-predicted VSK", fontsize=19, y=1.08)
    fig.tight_layout()
    fig.savefig(fig_dir / "mope_line_year_dumbbell.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "mope_line_year_dumbbell.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    table_dir = OUT / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT)
    df["year"] = df["year"].astype(str)
    summary = build_summary(df)
    summary.to_csv(table_dir / "mope_line_year_summary.csv", index=False)
    plot_line_year(summary)
    plot_line_year_i_shape(summary)
    plot_line_year_dumbbell(summary)
    print(summary.to_string(index=False))
    print(f"Saved outputs to {OUT}")


if __name__ == "__main__":
    main()
