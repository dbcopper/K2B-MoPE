import argparse
import csv
import copy
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import few_shot_cellpose_mope_ranking as base_pipeline

HEALTHY_DIR = ROOT / "examples" / "Healthy"
DISEASED_DIR = ROOT / "examples" / "Diseased"
DEFAULT_OUTPUT = ROOT / "output_results" / "ResNet18_Seed_Transfer"
DEFAULT_IMAGES = [
    ROOT / "discussion_dataset" / "test.jpg",
    ROOT / "discussion_dataset" / "IMG_6147.jpg",
]

CLASS_NAMES = ["Healthy", "Diseased"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SeedExampleDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image)
        return tensor, torch.tensor(label, dtype=torch.long), str(path)


def collect_labeled_examples():
    samples = []
    for path in sorted(HEALTHY_DIR.glob("*")):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            samples.append((path, 0))
    for path in sorted(DISEASED_DIR.glob("*")):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            samples.append((path, 1))
    return samples


def build_model(num_classes=2):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model, weights


def run_epoch(model, loader, criterion, optimizer, device, train_mode):
    model.train(train_mode)
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels, _paths in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(train_mode):
            logits = model(images)
            loss = criterion(logits, labels)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        preds = torch.argmax(logits, dim=1)
        total_loss += float(loss.item()) * images.size(0)
        total_correct += int((preds == labels).sum().item())
        total_count += int(images.size(0))

    return {
        "loss": total_loss / max(total_count, 1),
        "acc": total_correct / max(total_count, 1),
    }


def compute_classification_metrics(model, loader, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            for idx in range(images.size(0)):
                rows.append(
                    {
                        "path": paths[idx],
                        "true_label": int(labels[idx].item()),
                        "pred_label": int(preds[idx].item()),
                        "prob_healthy": float(probs[idx, 0].item()),
                        "prob_diseased": float(probs[idx, 1].item()),
                    }
                )
    return rows


def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def draw_prediction_overlay(image_bgr, records):
    overlay = image_bgr.copy()
    colors = {0: (0, 180, 0), 1: (0, 0, 220)}
    for record in records:
        mask = record["mask"]
        pred = int(record["pred_label"])
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.array(colors[pred])).astype(np.uint8)
        x1, y1, x2, y2 = record["bbox"]
        text = f"{CLASS_NAMES[pred][0]}:{record['prob_diseased']:.2f}"
        cv2.putText(
            overlay,
            text,
            (x1, max(20, y1 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return overlay


def load_or_segment_instances(image_bgr: np.ndarray, image_name: str, cache_dir: Path) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    cache_path = cache_dir / (
        f"cellpose_betterroi_d{base_pipeline.CELLPOSE_FIXED_DIAMETER}_"
        f"{image_name}_{gray.shape[0]}x{gray.shape[1]}.npz"
    )
    if cache_path.exists():
        return np.load(cache_path)["masks"].astype(np.uint32)

    rois = base_pipeline.build_candidate_rois_better(gray)
    if not rois:
        masks = np.zeros(gray.shape, dtype=np.uint32)
        np.savez_compressed(cache_path, masks=masks)
        return masks

    use_gpu = torch.cuda.is_available()
    from cellpose import models

    try:
        cellpose_model = models.CellposeModel(gpu=use_gpu)
    except Exception:
        cellpose_model = models.Cellpose(gpu=use_gpu, model_type=base_pipeline.CELLPOSE_MODEL)

    masks = base_pipeline.stitch_roi_masks(
        cellpose_model,
        gray,
        rois,
        diameter=base_pipeline.CELLPOSE_FIXED_DIAMETER,
        flow_threshold=base_pipeline.CELLPOSE_FLOW_THRESHOLD,
    )
    np.savez_compressed(cache_path, masks=masks)
    return masks


def save_class_crops(records, output_dir: Path):
    for class_id, class_name in enumerate(CLASS_NAMES):
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        class_records = sorted(
            [r for r in records if int(r["pred_label"]) == class_id],
            key=lambda r: r["prob_diseased"],
            reverse=True,
        )
        for rank, record in enumerate(class_records, start=1):
            name = (
                f"{rank:03d}_seed_{record['seed_id']:04d}_"
                f"pd_{record['prob_diseased']:.3f}_area_{record['area']:05d}.png"
            )
            cv2.imwrite(str(class_dir / name), record["crop_bgr"])


def infer_on_image(image_path: Path, model, preprocess, device, output_dir: Path):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    image_output = output_dir / image_path.stem
    image_output.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    masks = load_or_segment_instances(image_bgr, image_path.stem, cache_dir)
    rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    records = []
    model.eval()
    with torch.no_grad():
        for seed_id in range(1, int(masks.max()) + 1):
            mask = masks == seed_id
            area = int(mask.sum())
            if not (100 < area < 10000):
                continue
            ys, xs = np.where(mask)
            y1, y2 = int(ys.min()), int(ys.max())
            x1, x2 = int(xs.min()), int(xs.max())

            seed_rgb = rgb_image[y1:y2 + 1, x1:x2 + 1].copy()
            seed_mask = mask[y1:y2 + 1, x1:x2 + 1].copy()
            seed_rgb[~seed_mask] = [255, 255, 255]
            crop_bgr = cv2.cvtColor(seed_rgb, cv2.COLOR_RGB2BGR)

            tensor = preprocess(Image.fromarray(seed_rgb)).unsqueeze(0).to(device)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_label = int(np.argmax(probs))

            records.append(
                {
                    "seed_id": seed_id,
                    "area": area,
                    "bbox": (x1, y1, x2, y2),
                    "mask": mask,
                    "pred_label": pred_label,
                    "prob_healthy": float(probs[0]),
                    "prob_diseased": float(probs[1]),
                    "crop_bgr": crop_bgr,
                }
            )

    overlay = draw_prediction_overlay(image_bgr, records)
    cv2.imwrite(str(image_output / "prediction_overlay.png"), overlay)
    save_class_crops(records, image_output / "seed_crops")

    with open(image_output / "seed_predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "seed_id",
                "area",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "pred_label",
                "pred_name",
                "prob_healthy",
                "prob_diseased",
            ]
        )
        for record in records:
            x1, y1, x2, y2 = record["bbox"]
            writer.writerow(
                [
                    record["seed_id"],
                    record["area"],
                    x1,
                    y1,
                    x2,
                    y2,
                    record["pred_label"],
                    CLASS_NAMES[record["pred_label"]],
                    f"{record['prob_healthy']:.6f}",
                    f"{record['prob_diseased']:.6f}",
                ]
            )

    diseased_count = sum(int(r["pred_label"] == 1) for r in records)
    summary = {
        "image": str(image_path),
        "num_instances": len(records),
        "pred_healthy": len(records) - diseased_count,
        "pred_diseased": diseased_count,
        "mean_prob_diseased": float(np.mean([r["prob_diseased"] for r in records])) if records else 0.0,
    }
    save_json(summary, image_output / "summary.json")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Transfer-learning binary seed classifier with pretrained ResNet18.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--images", nargs="*", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_labeled_examples()
    labels = [label for _path, label in samples]
    train_samples, val_samples = train_test_split(
        samples,
        test_size=0.2,
        random_state=args.seed,
        stratify=labels,
    )

    model, weights = build_model()
    train_transform = weights.transforms()
    val_transform = weights.transforms()

    train_ds = SeedExampleDataset(train_samples, train_transform)
    val_ds = SeedExampleDataset(val_samples, val_transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_counts = np.bincount([label for _path, label in train_samples], minlength=2).astype(np.float32)
    class_weights = train_counts.sum() / np.maximum(train_counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, train_loader, criterion, optimizer, device, train_mode=True)
        val_stats = run_epoch(model, val_loader, criterion, optimizer, device, train_mode=False)
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_acc": train_stats["acc"],
            "val_loss": val_stats["loss"],
            "val_acc": val_stats["acc"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_stats['loss']:.4f} train_acc={train_stats['acc']:.4f} | "
            f"val_loss={val_stats['loss']:.4f} val_acc={val_stats['acc']:.4f}"
        )
        if val_stats["acc"] > best_val_acc:
            best_val_acc = val_stats["acc"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "history": history,
            "best_val_acc": best_val_acc,
            "seed": args.seed,
        },
        output_dir / "resnet18_seed_binary.pth",
    )
    save_json(history, output_dir / "training_history.json")

    val_rows = compute_classification_metrics(model, val_loader, device)
    save_json(val_rows, output_dir / "validation_predictions.json")

    summaries = []
    preprocess = weights.transforms()
    for image_path in args.images:
        summaries.append(infer_on_image(image_path, model, preprocess, device, output_dir / "inference"))
    save_json(summaries, output_dir / "inference_summary.json")

    train_info = {
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "train_class_counts": {
            CLASS_NAMES[i]: int(np.sum([label == i for _path, label in train_samples]))
            for i in range(2)
        },
        "val_class_counts": {
            CLASS_NAMES[i]: int(np.sum([label == i for _path, label in val_samples]))
            for i in range(2)
        },
        "best_val_acc": best_val_acc,
    }
    save_json(train_info, output_dir / "train_info.json")
    print(json.dumps(train_info, indent=2))


if __name__ == "__main__":
    main()
