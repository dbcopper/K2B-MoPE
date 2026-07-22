from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import few_shot_cellpose_mope_ranking as mope
import train_transfer_seed_classifier as resnet_seed

PNG_ROOT = ROOT / "data" / "seeds_png"
HEALTH_DIR = PNG_ROOT / "health"
DISEASE_DIR = PNG_ROOT / "disease"
TEST_DIR = PNG_ROOT / "test"
OUT_ROOT = ROOT / "output_results" / "current_seed_models"
TRAIN_EXAMPLES_DIR = OUT_ROOT / "training_examples"
TRAIN_HEALTHY_DIR = TRAIN_EXAMPLES_DIR / "Healthy"
TRAIN_DISEASED_DIR = TRAIN_EXAMPLES_DIR / "Diseased"
EXTRACT_CACHE_DIR = OUT_ROOT / "seedpose_cache"
MOPE_OUT = OUT_ROOT / "MoPE"
RESNET_OUT = OUT_ROOT / "ResNet18"


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def valid_area(area: int) -> bool:
    return 100 < area < 10000


def save_kernel_crop(
    rgb_image: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    dst_path: Path,
) -> None:
    x1, y1, x2, y2 = bbox
    crop_rgb = rgb_image[y1:y2 + 1, x1:x2 + 1].copy()
    crop_mask = mask[y1:y2 + 1, x1:x2 + 1].copy()
    crop_rgb[~crop_mask] = [0, 0, 0]
    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(dst_path), crop_bgr)


def segment_image(image_path: Path, cache_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    masks = resnet_seed.load_or_segment_instances(image_bgr, image_path.stem, cache_dir)
    return rgb, gray, masks


def extract_kernels_from_dir(src_dir: Path, dst_dir: Path, cache_dir: Path) -> dict:
    dst_dir.mkdir(parents=True, exist_ok=True)
    summary = {"source_dir": str(src_dir), "images": [], "num_kernels": 0}

    for image_path in sorted(src_dir.glob("*.png")):
        rgb, _gray, masks = segment_image(image_path, cache_dir)
        kept = 0
        for seed_id in range(1, int(masks.max()) + 1):
            mask = masks == seed_id
            area = int(mask.sum())
            if not valid_area(area):
                continue
            ys, xs = np.where(mask)
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            dst_path = dst_dir / f"{image_path.stem}_seed_{seed_id:04d}.png"
            save_kernel_crop(rgb, mask, bbox, dst_path)
            kept += 1
        summary["images"].append({"image": image_path.name, "kept_kernels": kept})
        summary["num_kernels"] += kept
        print(f"Extracted {kept} kernels from {image_path.name} -> {dst_dir.name}")
    return summary


def train_mope_new_only() -> dict:
    healthy_feats, feature_names = mope.extract_features_from_dir(str(TRAIN_HEALTHY_DIR))
    diseased_feats, _ = mope.extract_features_from_dir(str(TRAIN_DISEASED_DIR))
    healthy_norm, diseased_norm, feat_mean, feat_std = mope.normalize_features(healthy_feats, diseased_feats)

    model = mope.MoPERanker()
    trainer = mope.PairwiseRankingTrainer(
        model,
        lr=mope.LEARNING_RATE,
        weight_decay=mope.WEIGHT_DECAY,
        margin=mope.RANKING_MARGIN,
    )
    history = trainer.train(healthy_norm, diseased_norm, num_epochs=mope.NUM_EPOCHS, verbose=True)

    calibrator = mope.SeverityCalibrator(model)
    thresholds = calibrator.calibrate(healthy_norm, diseased_norm)

    MOPE_OUT.mkdir(parents=True, exist_ok=True)
    stages_dir = MOPE_OUT / "stages"
    stages_dir.mkdir(parents=True, exist_ok=True)
    mope.plot_training_history(history, str(stages_dir / "training_history.png"))

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feat_mean": feat_mean,
            "feat_std": feat_std,
            "thresholds": thresholds,
            "calibrator_h_stats": calibrator.h_stats,
            "calibrator_d_stats": calibrator.d_stats,
            "feature_names": feature_names,
            "history": history,
            "loo_binary_acc": None,
            "loo_ranking_acc": None,
            "config": {
                "color_dim": len(mope.COLOR_INDICES),
                "texture_dim": len(mope.TEXTURE_INDICES),
                "morph_dim": len(mope.MORPH_INDICES),
                "ranking_margin": mope.RANKING_MARGIN,
            },
        },
        MOPE_OUT / "mope_model.pth",
    )

    return {
        "n_healthy": int(len(healthy_feats)),
        "n_diseased": int(len(diseased_feats)),
        "feature_dim": int(len(feature_names)),
        "thresholds": thresholds,
        "healthy_score_mean": float(calibrator.h_stats["mean"]),
        "diseased_score_mean": float(calibrator.d_stats["mean"]),
    }


def infer_mope_test() -> list[dict]:
    loaded = mope.load_model(str(MOPE_OUT))
    if loaded is None:
        raise RuntimeError("Failed to load newly trained MoPE model.")
    model, calibrator, feat_mean, feat_std, feature_names, _history, _loo_bin, _loo_rank = loaded

    results = []
    for image_path in sorted(TEST_DIR.glob("*.png")):
        print(f"MoPE inference: {image_path.name}")
        out = mope.process_image(
            str(image_path),
            model,
            calibrator,
            feat_mean,
            feat_std,
            feature_names,
            str(MOPE_OUT),
        )
        if out is not None:
            score_arr = np.array([r["severity_score"] for r in out["results"]], dtype=np.float64)
            train_h_n = calibrator.h_stats.get("n", 1)
            train_d_n = calibrator.d_stats.get("n", 1)
            ref_mean = (
                calibrator.h_stats["mean"] * train_h_n + calibrator.d_stats["mean"] * train_d_n
            ) / (train_h_n + train_d_n)
            vsk_threshold = calibrator.d_stats["mean"] + 0.25 * (float(score_arr.mean()) - ref_mean)
            predicted_vsk_pct = float(np.mean(score_arr >= vsk_threshold) * 100.0)
            results.append(
                {
                    "image_name": image_path.name,
                    "n_kernels": int(len(out["results"])),
                    "mean_score": float(score_arr.mean()),
                    "vsk_threshold": float(vsk_threshold),
                    "predicted_vsk_pct": predicted_vsk_pct,
                }
            )
    return results


def train_resnet_new_only() -> dict:
    all_samples = []
    for path in sorted(TRAIN_HEALTHY_DIR.glob("*.png")):
        all_samples.append((path, 0))
    for path in sorted(TRAIN_DISEASED_DIR.glob("*.png")):
        all_samples.append((path, 1))

    labels = [label for _path, label in all_samples]
    train_samples, val_samples = resnet_seed.train_test_split(
        all_samples,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    model, weights = resnet_seed.build_model()
    train_transform = weights.transforms()
    val_transform = weights.transforms()
    train_ds = resnet_seed.SeedExampleDataset(train_samples, train_transform)
    val_ds = resnet_seed.SeedExampleDataset(val_samples, val_transform)
    train_loader = resnet_seed.DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = resnet_seed.DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_counts = np.bincount([label for _path, label in train_samples], minlength=2).astype(np.float32)
    class_weights = train_counts.sum() / np.maximum(train_counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    history = []
    best_state = model.state_dict()
    best_val_acc = -1.0
    for epoch in range(1, 16 + 1):
        train_stats = resnet_seed.run_epoch(model, train_loader, criterion, optimizer, device, train_mode=True)
        val_stats = resnet_seed.run_epoch(model, val_loader, criterion, optimizer, device, train_mode=False)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_acc": train_stats["acc"],
                "val_loss": val_stats["loss"],
                "val_acc": val_stats["acc"],
            }
        )
        print(
            f"ResNet18 epoch {epoch:02d}/16 | "
            f"train_acc={train_stats['acc']:.4f} | val_acc={val_stats['acc']:.4f}"
        )
        if val_stats["acc"] > best_val_acc:
            best_val_acc = val_stats["acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    RESNET_OUT.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": resnet_seed.CLASS_NAMES,
            "history": history,
            "best_val_acc": best_val_acc,
            "seed": 42,
        },
        RESNET_OUT / "resnet18_seed_binary.pth",
    )
    (RESNET_OUT / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    val_rows = resnet_seed.compute_classification_metrics(model, val_loader, device)
    (RESNET_OUT / "validation_predictions.json").write_text(json.dumps(val_rows, indent=2), encoding="utf-8")

    train_info = {
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "train_class_counts": {
            resnet_seed.CLASS_NAMES[i]: int(np.sum([label == i for _path, label in train_samples]))
            for i in range(2)
        },
        "val_class_counts": {
            resnet_seed.CLASS_NAMES[i]: int(np.sum([label == i for _path, label in val_samples]))
            for i in range(2)
        },
        "best_val_acc": float(best_val_acc),
    }
    (RESNET_OUT / "train_info.json").write_text(json.dumps(train_info, indent=2), encoding="utf-8")
    return train_info


def infer_resnet_test() -> list[dict]:
    checkpoint = torch.load(RESNET_OUT / "resnet18_seed_binary.pth", map_location="cpu", weights_only=False)
    model, weights = resnet_seed.build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    preprocess = weights.transforms()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    inference_root = RESNET_OUT / "inference"
    inference_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for image_path in sorted(TEST_DIR.glob("*.png")):
        print(f"ResNet18 inference: {image_path.name}")
        summaries.append(
            resnet_seed.infer_on_image(image_path, model, preprocess, device, inference_root)
        )
    (RESNET_OUT / "inference_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return summaries


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    reset_dir(TRAIN_EXAMPLES_DIR)
    reset_dir(MOPE_OUT)
    reset_dir(RESNET_OUT)
    EXTRACT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    health_extract = extract_kernels_from_dir(HEALTH_DIR, TRAIN_HEALTHY_DIR, EXTRACT_CACHE_DIR)
    disease_extract = extract_kernels_from_dir(DISEASE_DIR, TRAIN_DISEASED_DIR, EXTRACT_CACHE_DIR)

    mope_train = train_mope_new_only()
    mope_test = infer_mope_test()

    resnet_train = train_resnet_new_only()
    resnet_test = infer_resnet_test()

    summary = {
        "source_png_root": str(PNG_ROOT),
        "output_root": str(OUT_ROOT),
        "new_training_only": True,
        "extraction": {
            "healthy": health_extract,
            "diseased": disease_extract,
        },
        "mope_train": mope_train,
        "mope_test": mope_test,
        "resnet_train": resnet_train,
        "resnet_test": resnet_test,
    }
    (OUT_ROOT / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
