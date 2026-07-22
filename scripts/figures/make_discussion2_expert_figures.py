from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "discussions" / "discussion2"
KERNEL_PRED = ROOT / "results" / "result2" / "per_kernel" / "table8_per_kernel_predictions.csv"
IMAGE_ROOT = ROOT / "dataset" / "test"
MASK_CACHE_ROOT = ROOT / "results" / "result2" / "seedpose_cache_dia95" / "mask_cache" / "seedpose_cyto"

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

IMAGE_EXTENSIONS = [".DNG", ".dng", ".jpg", ".jpeg", ".png", ".tif", ".tiff"]
EXPERT_LABELS = ["Color", "Texture", "Morphology"]
GATE_COLS = ["mope_gate_color", "mope_gate_texture", "mope_gate_morph"]
EXPERT_COLS = ["mope_expert_color", "mope_expert_texture", "mope_expert_morph"]
COLORS = ["#D55E00", "#0072B2", "#009E73"]


def read_rgb(path: Path) -> np.ndarray:
    arr = iio.imread(path)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr))
        if hi > lo:
            arr = (255.0 * (arr - lo) / (hi - lo)).clip(0, 255)
        arr = arr.astype(np.uint8)
    return arr


def image_path(year: str, plot: str) -> Path:
    year_dir = IMAGE_ROOT / year
    for suffix in IMAGE_EXTENSIONS:
        candidate = year_dir / f"{plot}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing image for year={year}, plot={plot}")


def load_label(year: str, plot: str) -> np.ndarray:
    data = np.load(MASK_CACHE_ROOT / year / f"{plot}.npz", allow_pickle=True)
    return data["label"]


def crop_kernel(image: np.ndarray, label: np.ndarray, row: pd.Series, size: int = 150) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(int, [row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]])
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad = int(max(bw, bh) * 0.32)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    half = max(bw, bh) // 2 + pad
    xa, xb = max(0, cx - half), min(w, cx + half)
    ya, yb = max(0, cy - half), min(h, cy + half)
    crop = image[ya:yb, xa:xb].copy()
    mask_crop = label[ya:yb, xa:xb] == int(row["seed_id"])
    if mask_crop.any():
        crop[~mask_crop] = 255
    hh, ww = crop.shape[:2]
    canvas_size = max(hh, ww)
    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
    yo = (canvas_size - hh) // 2
    xo = (canvas_size - ww) // 2
    canvas[yo : yo + hh, xo : xo + ww] = crop
    return np.asarray(Image.fromarray(canvas).resize((size, size), Image.Resampling.LANCZOS))


def displayable(df: pd.DataFrame) -> pd.Series:
    width = (df["bbox_x2"] - df["bbox_x1"]).clip(lower=1)
    height = (df["bbox_y2"] - df["bbox_y1"]).clip(lower=1)
    aspect = np.maximum(width / height, height / width)
    fill = df["area"] / (width * height)
    return (df["area"] >= 3000) & (aspect <= 3.2) & (fill >= 0.25)


def plot_expert_trends(df: pd.DataFrame) -> None:
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    work = df.dropna(subset=["mope_score", *GATE_COLS, *EXPERT_COLS]).copy()
    work = work[displayable(work)]
    work["score_bin"] = pd.qcut(work["mope_score"], q=10, duplicates="drop")
    grouped = work.groupby("score_bin", observed=True)
    centers = grouped["mope_score"].mean().to_numpy()
    gates = grouped[GATE_COLS].mean()
    experts = grouped[EXPERT_COLS].mean()

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), sharex=True)
    for i, label in enumerate(EXPERT_LABELS):
        axes[0].plot(centers, gates.iloc[:, i], marker="o", linewidth=2.2, color=COLORS[i], label=label)
        axes[1].plot(centers, experts.iloc[:, i], marker="o", linewidth=2.2, color=COLORS[i], label=label)

    axes[0].set_title("Adaptive gating weights", fontsize=17)
    axes[1].set_title("Expert severity scores", fontsize=17)
    for ax in axes:
        ax.set_xlabel("MoPE kernel severity score", fontsize=15)
        ax.tick_params(labelsize=13)
        ax.grid(alpha=0.25)
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("Mean gating weight", fontsize=15)
    axes[1].set_ylabel("Mean expert score", fontsize=15)
    axes[0].legend(frameon=False, fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / "mope_expert_trends.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "mope_expert_trends.pdf", bbox_inches="tight")
    plt.close(fig)


def select_cases(df: pd.DataFrame) -> pd.DataFrame:
    work = df.dropna(subset=["mope_score", *GATE_COLS, *EXPERT_COLS]).copy()
    work = work[displayable(work)].copy()
    work["max_gate"] = work[GATE_COLS].max(axis=1)
    work["gate_entropy"] = -(
        work[GATE_COLS].clip(lower=1e-8) * np.log(work[GATE_COLS].clip(lower=1e-8))
    ).sum(axis=1)
    cases = []
    specs = [
        ("Low severity", work["mope_score"].between(0.20, 0.35) & (work["max_gate"] <= 0.75)),
        (
            "Color-elevated",
            work["mope_score"].between(0.45, 0.75)
            & (work["mope_gate_color"] == work[GATE_COLS].max(axis=1))
            & (work["max_gate"].between(0.45, 0.85)),
        ),
        (
            "Texture-elevated",
            work["mope_score"].between(0.45, 0.85)
            & (work["mope_gate_texture"] == work[GATE_COLS].max(axis=1))
            & (work["max_gate"].between(0.45, 0.85)),
        ),
        ("High severity", work["mope_score"].between(0.80, 0.98) & (work["max_gate"] <= 0.85)),
    ]
    for label, mask in specs:
        pool = work[mask].copy()
        if pool.empty:
            pool = work[work["max_gate"] <= 0.90].copy()
        if pool.empty:
            pool = work.copy()
        if label == "Low severity":
            target = 0.28
            idx = ((pool["mope_score"] - target).abs() - 0.08 * pool["gate_entropy"]).idxmin()
        elif label == "High severity":
            target = 0.88
            idx = ((pool["mope_score"] - target).abs() - 0.08 * pool["gate_entropy"]).idxmin()
        else:
            target = 0.62
            idx = ((pool["mope_score"] - target).abs() + (pool["max_gate"] - 0.65).abs()).idxmin()
        row = work.loc[idx].copy()
        row["case_label"] = label
        cases.append(row)
    return pd.DataFrame(cases)


def plot_case_study(df: pd.DataFrame) -> None:
    fig_dir = OUT / "figures"
    table_dir = OUT / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    cases = select_cases(df)
    cases.to_csv(table_dir / "mope_expert_case_study_kernels.csv", index=False)

    image_cache: dict[tuple[str, str], np.ndarray] = {}
    label_cache: dict[tuple[str, str], np.ndarray] = {}
    fig, axes = plt.subplots(len(cases), 3, figsize=(9.5, 9.4), gridspec_kw={"width_ratios": [1.1, 1.45, 1.45]})

    for i, (_, row) in enumerate(cases.iterrows()):
        key = (str(row["year"]), str(row["plot"]))
        if key not in image_cache:
            image_cache[key] = read_rgb(image_path(*key))
            label_cache[key] = load_label(*key)
        crop = crop_kernel(image_cache[key], label_cache[key], row)

        ax_img, ax_gate, ax_exp = axes[i]
        ax_img.imshow(crop)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        ax_img.set_title(
            f"{row['case_label']}\nscore={float(row['mope_score']):.2f}",
            fontsize=13,
        )
        for spine in ax_img.spines.values():
            spine.set_linewidth(2.2)
            spine.set_edgecolor(plt.cm.RdYlGn_r(float(row["mope_score"])))

        gate_vals = [float(row[c]) for c in GATE_COLS]
        exp_vals = [float(row[c]) for c in EXPERT_COLS]
        ax_gate.bar(EXPERT_LABELS, gate_vals, color=COLORS, alpha=0.85)
        ax_exp.bar(EXPERT_LABELS, exp_vals, color=COLORS, alpha=0.85)
        for ax, vals, ylabel in [(ax_gate, gate_vals, "Gating weight"), (ax_exp, exp_vals, "Expert score")]:
            ax.set_ylim(0, 1.05)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.tick_params(axis="x", labelrotation=25, labelsize=11)
            ax.tick_params(axis="y", labelsize=10)
            ax.grid(axis="y", alpha=0.2)
            for j, v in enumerate(vals):
                ax.text(j, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(fig_dir / "mope_expert_case_study.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "mope_expert_case_study.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    kernel_df = pd.read_csv(KERNEL_PRED, dtype={"year": str, "plot": str})

    plot_expert_trends(kernel_df)
    plot_case_study(kernel_df)

    print(f"Saved discussion2 outputs to {OUT}")


if __name__ == "__main__":
    main()
