from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "discussions" / "discussion2"
MODEL_PATH = ROOT / "results" / "result2" / "models" / "mope" / "mope_model.pth"

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False


def load_weights() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    feature_names = list(checkpoint["feature_names"])

    color_names = feature_names[:11]
    texture_names = feature_names[11:22]
    morph_names = feature_names[22:25]

    color_w = state["expert_color.fc.weight"].detach().cpu().numpy().reshape(-1)
    texture_w = state["expert_texture.fc.weight"].detach().cpu().numpy().reshape(-1)
    morph_w = state["expert_morph.fc.weight"].detach().cpu().numpy().reshape(-1)

    return (
        pd.DataFrame({"expert": "Color", "feature": color_names, "weight": color_w}),
        pd.DataFrame({"expert": "Texture", "feature": texture_names, "weight": texture_w}),
        pd.DataFrame({"expert": "Morphology", "feature": morph_names, "weight": morph_w}),
    )


def plot_feature_weights(frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    fig_dir = OUT / "figures"
    table_dir = OUT / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    all_weights = pd.concat(frames, ignore_index=True)
    all_weights.to_csv(table_dir / "mope_feature_weights.csv", index=False)

    titles = ["Color Expert", "Texture Expert", "Morphology Expert"]
    colors = ["#E64B35", "#4DBBD5", "#B8C0C4"]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8))

    for ax, frame, title, color in zip(axes, frames, titles, colors):
        frame = frame.copy()
        frame["abs_weight"] = frame["weight"].abs()
        frame = frame.sort_values("weight", ascending=True)
        bar_colors = [color if value >= 0 else "#C5CDD0" for value in frame["weight"]]
        ax.barh(frame["feature"], frame["weight"], color=bar_colors, alpha=0.92)
        ax.axvline(0, color="#333333", linewidth=0.9)
        ax.set_title(f"{title}\nLearned weights", fontsize=15)
        ax.set_xlabel("Weight (+ = higher severity)", fontsize=13)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(axis="x", alpha=0.22)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    xlim = max(abs(all_weights["weight"].min()), abs(all_weights["weight"].max())) * 1.12
    for ax in axes:
        ax.set_xlim(-xlim, xlim)

    fig.tight_layout(w_pad=2.5)
    fig.savefig(fig_dir / "mope_feature_weights.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "mope_feature_weights.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    frames = load_weights()
    plot_feature_weights(frames)
    print(f"Saved feature-weight figure to {OUT / 'figures'}")


if __name__ == "__main__":
    main()
