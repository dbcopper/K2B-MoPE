from __future__ import annotations

import sys
from pathlib import Path

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "discussions" / "discussion1"
KERNEL_PRED = ROOT / "results" / "result2" / "per_kernel" / "table8_per_kernel_predictions.csv"
IMAGE_ROOT = ROOT / "dataset" / "test"
MASK_CACHE_ROOT = ROOT / "results" / "result2" / "seedpose_cache_dia95" / "mask_cache" / "seedpose_cyto"

IMAGE_EXTENSIONS = [".DNG", ".dng", ".jpg", ".jpeg", ".png", ".tif", ".tiff"]


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
    path = MASK_CACHE_ROOT / year / f"{plot}.npz"
    data = np.load(path, allow_pickle=True)
    return data["label"]


def crop_kernel(image: np.ndarray, label: np.ndarray, row: pd.Series) -> np.ndarray:
    h, w = image.shape[:2]
    x1 = int(row["bbox_x1"])
    y1 = int(row["bbox_y1"])
    x2 = int(row["bbox_x2"])
    y2 = int(row["bbox_y2"])
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad = int(max(bw, bh) * 0.25)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    half = max(bw, bh) // 2 + pad
    xa = max(0, cx - half)
    xb = min(w, cx + half)
    ya = max(0, cy - half)
    yb = min(h, cy + half)
    crop = image[ya:yb, xa:xb].copy()
    mask_crop = label[ya:yb, xa:xb] == int(row["seed_id"])
    if crop.size == 0:
        crop = np.zeros((96, 96, 3), dtype=np.uint8)
        return crop
    if mask_crop.shape[:2] == crop.shape[:2] and mask_crop.any():
        crop[~mask_crop] = 255
    return crop


def is_displayable(df: pd.DataFrame) -> pd.Series:
    width = (df["bbox_x2"] - df["bbox_x1"]).clip(lower=1)
    height = (df["bbox_y2"] - df["bbox_y1"]).clip(lower=1)
    aspect = np.maximum(width / height, height / width)
    fill_ratio = df["area"] / (width * height)
    return (df["area"] >= 3000) & (aspect <= 3.2) & (fill_ratio >= 0.25)


def quantile_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    ranked = df[is_displayable(df)].sort_values("mope_score").reset_index(drop=True)
    if len(ranked) < n:
        raise ValueError(f"Only {len(ranked)} displayable kernels available, requested {n}")
    idx = np.linspace(0, len(ranked) - 1, n).round().astype(int)
    return ranked.iloc[idx].copy().reset_index(drop=True)


def square_canvas(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape[:2]
    size = max(h, w)
    canvas = np.ones((size, size, 3), dtype=np.uint8) * 255
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    canvas[y_off : y_off + h, x_off : x_off + w] = crop
    return canvas


def make_strip(n_kernels: int = 48) -> None:
    OUT.joinpath("figures").mkdir(parents=True, exist_ok=True)
    OUT.joinpath("tables").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(KERNEL_PRED)
    df["year"] = df["year"].astype(str)
    df["plot"] = df["plot"].astype(str)
    df = df.dropna(subset=["mope_score", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"])
    sample = quantile_sample(df, n_kernels)
    sample.to_csv(OUT / "tables" / "mope_ranking_strip_kernels.csv", index=False)

    image_cache: dict[tuple[str, str], np.ndarray] = {}
    label_cache: dict[tuple[str, str], np.ndarray] = {}
    crops = []
    for _, row in sample.iterrows():
        key = (str(row["year"]), str(row["plot"]))
        if key not in image_cache:
            image_cache[key] = read_rgb(image_path(*key))
            label_cache[key] = load_label(*key)
        crops.append(crop_kernel(image_cache[key], label_cache[key], row))

    n = len(crops)
    ncols = min(n, 16)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.2, nrows * 1.6))
    if nrows == 1:
        axes = axes.reshape(1, -1) if n > 1 else np.array([[axes]])
    cmap = plt.get_cmap("RdYlGn_r")
    scores = sample["mope_score"].to_numpy(dtype=float)
    norm = plt.Normalize(vmin=0.0, vmax=1.0)

    for i in range(nrows * ncols):
        row_i, col_i = divmod(i, ncols)
        ax = axes[row_i, col_i]
        if i < n:
            score = float(scores[i])
            ax.imshow(square_canvas(crops[i]))
            for spine in ax.spines.values():
                spine.set_linewidth(3.0)
                spine.set_edgecolor(cmap(norm(score)))
            ax.set_xlabel(f"{score:.2f}", fontsize=13, color="black", fontweight="bold", labelpad=6)
        else:
            ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.02, pad=0.02, aspect=30)
    cbar.set_label("Severity Score", fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    out_png = OUT / "figures" / "mope_ranking_strip.png"
    out_pdf = OUT / "figures" / "mope_ranking_strip.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved ranking strip to {out_png}")
    print(
        "score range:",
        f"{scores.min():.3f}",
        "to",
        f"{scores.max():.3f}",
        "n=",
        len(scores),
    )


if __name__ == "__main__":
    try:
        make_strip()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
