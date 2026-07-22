from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = ROOT / "results" / "result5_method_comparison" / "per_plot_predictions_extended.csv"
OUT_DIR = ROOT / "results" / "result5_method_comparison"
COPY_DIR = ROOT / "results" / "result2" / "figures"


MODELS = [
    ("SVM (RBF)", "svm_standard", "#9aa3b2", "o", 0.48, 40),
    ("Random Forest", "rf_standard", "#2a9d8f", "s", 0.54, 42),
    ("RankSVM", "ranksvm", "#7b5ea7", "v", 0.60, 44),
    ("ResNet18", "resnet18_standard", "#457b9d", "^", 0.54, 42),
    ("ViT", "vit_standard", "#f4a261", "D", 0.50, 42),
    ("MoPE", "mope", "#d62828", "P", 0.76, 54),
]


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    COPY_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    y_true = df["manual_vsk_pct"].astype(float)

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )
    fig, ax = plt.subplots(figsize=(6.0, 6.0))

    for label, column, color, marker, alpha, size in MODELS:
        ax.scatter(
            y_true,
            df[column].astype(float),
            s=size,
            alpha=alpha,
            color=color,
            marker=marker,
            edgecolors="white",
            linewidths=0.35,
            label=label,
        )

    ax.plot([0, 100], [0, 100], linestyle="--", color="#3a3a3a", linewidth=1.2, label="y = x")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Manual VSK (%)")
    ax.set_ylabel("Predicted VSK (%)")
    ax.set_title("Manual vs. Predicted VSK Across Models", fontweight="bold")
    ax.grid(True, alpha=0.18)
    ax.legend(loc="lower right", frameon=True, fontsize=9, markerscale=0.9, borderpad=0.7)
    fig.tight_layout()

    for stem_dir in [OUT_DIR, COPY_DIR]:
        fig.savefig(stem_dir / "manual_vs_predicted_overlay_with_ranksvm.png", dpi=300, bbox_inches="tight")
        fig.savefig(stem_dir / "manual_vs_predicted_overlay_with_ranksvm.pdf", bbox_inches="tight")
    plt.close(fig)

    print(OUT_DIR / "manual_vs_predicted_overlay_with_ranksvm.png")
    print(COPY_DIR / "manual_vs_predicted_overlay_with_ranksvm.png")


if __name__ == "__main__":
    main()
