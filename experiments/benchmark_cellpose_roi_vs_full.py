import argparse
import csv
import glob
import os
import statistics
import time

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


CELLPOSE_MODEL = "cyto"
DEFAULT_IMAGES = ["./origin.jpg"]
DEFAULT_OUTPUT_DIR = os.path.join("output_results", "benchmarks")
def build_ablation_methods(diameter_values=None):
    methods = [
        {
            "name": "A0_full_image_auto",
            "label": "A0 Full Image",
            "roi_mode": "full_image",
            "roi_detector": "none",
            "diameter_mode": "auto",
            "fixed_diameter": None,
        },
        {
            "name": "A2_multi_roi_otsu_auto",
            "label": "A2 Better ROI",
            "roi_mode": "split",
            "roi_detector": "downsample_otsu_cc",
            "diameter_mode": "auto",
            "fixed_diameter": None,
        },
    ]

    if diameter_values:
        for diameter in diameter_values:
            methods.append(
                {
                    "name": f"A3_multi_roi_otsu_fixed_{int(round(diameter))}",
                    "label": f"A3 Fixed D={int(round(diameter))}",
                    "roi_mode": "split",
                    "roi_detector": "downsample_otsu_cc",
                    "diameter_mode": "fixed",
                    "fixed_diameter": float(diameter),
                }
            )
    else:
        methods.append(
            {
                "name": "A3_multi_roi_otsu_fixed",
                "label": "A3 Better ROI + Fixed D",
                "roi_mode": "split",
                "roi_detector": "downsample_otsu_cc",
                "diameter_mode": "fixed",
                "fixed_diameter": None,
            }
        )

    return methods


def sync_device():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def sanitize_stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_image(path, image):
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, image)


def colorize_mask(masks):
    if masks.max() == 0:
        return np.zeros((masks.shape[0], masks.shape[1], 3), dtype=np.uint8)
    hue = ((masks.astype(np.uint64) * 53) % 180).astype(np.uint8)
    sat = np.where(masks > 0, 220, 0).astype(np.uint8)
    val = np.where(masks > 0, 255, 0).astype(np.uint8)
    hsv = np.stack([hue, sat, val], axis=-1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def overlay_mask_on_image(image, masks, alpha=0.45):
    color_mask = colorize_mask(masks)
    overlay = image.copy()
    foreground = masks > 0
    overlay[foreground] = cv2.addWeighted(
        image[foreground], 1.0 - alpha, color_mask[foreground], alpha, 0
    )
    return overlay


def draw_rois(image, rois, color=(0, 255, 255), thickness=3):
    canvas = image.copy()
    for idx, roi in enumerate(rois, start=1):
        x1, y1, x2, y2 = roi["bbox"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            canvas,
            str(idx),
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas


def estimate_diameter_from_rois(candidate_rois):
    if not candidate_rois:
        return None
    diameters = [
        np.sqrt(4 * roi.get("area", 0) / np.pi)
        for roi in candidate_rois
        if roi.get("area", 0) > 50
    ]
    return int(np.median(diameters)) if diameters else None


def build_candidate_rois_adaptive(gray, margin=30, min_area=100):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    binary_open = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1
    )
    binary_close = cv2.morphologyEx(
        binary_open, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2
    )
    contours, _ = cv2.findContours(binary_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidate_rois = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(gray.shape[1], x + w + margin)
        y2 = min(gray.shape[0], y + h + margin)
        candidate_rois.append({"bbox": (x1, y1, x2, y2), "area": area})

    merged_rois = merge_overlapping_rois(candidate_rois)
    debug = {
        "blurred": blurred,
        "binary": binary,
        "binary_open": binary_open,
        "binary_close": binary_close,
        "candidate_rois": candidate_rois,
        "merged_rois": merged_rois,
    }
    return merged_rois, debug


def estimate_kernel_diameter_from_image(
    gray,
    downsample_max_side=1600,
    min_area=80,
    max_area_fraction=0.02,
    min_solidity=0.65,
    min_aspect=0.35,
    max_aspect=3.2,
):
    """
    Estimate a robust global kernel diameter from a dense seed image.

    Strategy:
    1. Downsample large images for speed.
    2. Segment dark kernels from bright background with Otsu thresholding.
    3. Keep connected components that look like plausible single kernels.
    4. Use the median equivalent-circle diameter of those components.

    Returns:
        estimated_diameter (int | None), debug_info (dict)
    """
    h, w = gray.shape[:2]
    scale = 1.0
    longest_side = max(h, w)
    if longest_side > downsample_max_side:
        scale = downsample_max_side / float(longest_side)
        small = cv2.resize(
            gray,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = gray

    blurred = cv2.GaussianBlur(small, (5, 5), 0)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k3, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k5, iterations=1)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    image_area = float(binary.shape[0] * binary.shape[1])
    max_area = max_area_fraction * image_area

    diameters = []
    kept_components = 0

    for cid in range(1, n_labels):
        area = float(stats[cid, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue

        x = stats[cid, cv2.CC_STAT_LEFT]
        y = stats[cid, cv2.CC_STAT_TOP]
        bw = stats[cid, cv2.CC_STAT_WIDTH]
        bh = stats[cid, cv2.CC_STAT_HEIGHT]
        aspect = bw / float(max(bh, 1))
        if aspect < min_aspect or aspect > max_aspect:
            continue

        component_mask = (labels[y : y + bh, x : x + bw] == cid).astype(np.uint8)
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        hull_area = max(float(cv2.contourArea(hull)), 1.0)
        solidity = float(cv2.contourArea(cnt)) / hull_area
        if solidity < min_solidity:
            continue

        eq_diameter = np.sqrt(4.0 * area / np.pi)
        diameters.append(eq_diameter / scale)
        kept_components += 1

    if not diameters:
        return None, {
            "scale": scale,
            "num_components": int(n_labels - 1),
            "num_kept_components": 0,
            "binary": binary,
        }

    estimate = int(round(np.median(diameters)))
    return estimate, {
        "scale": scale,
        "num_components": int(n_labels - 1),
        "num_kept_components": kept_components,
        "binary": binary,
        "diameters": diameters,
    }


def build_candidate_rois_otsu_downsample(
    gray,
    margin=30,
    min_area=100,
    downsample_max_side=1600,
):
    h, w = gray.shape[:2]
    scale = 1.0
    longest_side = max(h, w)
    if longest_side > downsample_max_side:
        scale = downsample_max_side / float(longest_side)
        work = cv2.resize(
            gray,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        work = gray

    blurred = cv2.GaussianBlur(work, (5, 5), 0)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    binary_open = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1
    )
    binary_close = cv2.morphologyEx(
        binary_open, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1
    )
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_close, connectivity=8)

    candidate_rois = []
    for cid in range(1, n_labels):
        area = float(stats[cid, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[cid, cv2.CC_STAT_LEFT])
        y = int(stats[cid, cv2.CC_STAT_TOP])
        bw = int(stats[cid, cv2.CC_STAT_WIDTH])
        bh = int(stats[cid, cv2.CC_STAT_HEIGHT])

        x1 = max(0, int(np.floor((x - margin) / scale)))
        y1 = max(0, int(np.floor((y - margin) / scale)))
        x2 = min(gray.shape[1], int(np.ceil((x + bw + margin) / scale)))
        y2 = min(gray.shape[0], int(np.ceil((y + bh + margin) / scale)))
        candidate_rois.append({"bbox": (x1, y1, x2, y2), "area": area / (scale * scale)})

    merged_rois = merge_overlapping_rois(candidate_rois)
    debug = {
        "scale": scale,
        "blurred": blurred,
        "binary": binary,
        "binary_open": binary_open,
        "binary_close": binary_close,
        "candidate_rois": candidate_rois,
        "merged_rois": merged_rois,
    }
    return merged_rois, debug


def merge_overlapping_rois(rois):
    if not rois:
        return []

    def intersects(b1, b2):
        return not (b1[2] < b2[0] or b2[2] < b1[0] or b1[3] < b2[1] or b2[3] < b1[1])

    merged_groups = []
    used = [False] * len(rois)

    for i in range(len(rois)):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        j = 0
        while j < len(group):
            for k in range(len(rois)):
                if not used[k] and intersects(rois[group[j]]["bbox"], rois[k]["bbox"]):
                    group.append(k)
                    used[k] = True
            j += 1
        merged_groups.append(group)

    merged = []
    for group in merged_groups:
        bboxes = [rois[idx]["bbox"] for idx in group]
        merged.append(
            {
                "bbox": (
                    min(b[0] for b in bboxes),
                    min(b[1] for b in bboxes),
                    max(b[2] for b in bboxes),
                    max(b[3] for b in bboxes),
                ),
                "count": len(group),
            }
        )
    return merged


def build_candidate_rois(gray, detector="adaptive", margin=30, min_area=100):
    if detector == "adaptive":
        return build_candidate_rois_adaptive(gray, margin=margin, min_area=min_area)
    if detector == "downsample_otsu_cc":
        return build_candidate_rois_otsu_downsample(gray, margin=margin, min_area=min_area)
    raise ValueError(f"Unsupported ROI detector: {detector}")


def load_cellpose_model(use_gpu):
    from cellpose import models

    try:
        return models.CellposeModel(gpu=use_gpu)
    except Exception:
        return models.Cellpose(gpu=use_gpu, model_type=CELLPOSE_MODEL)


def evaluate_cellpose(model, image, diameter, flow_threshold):
    sync_device()
    start = time.perf_counter()
    masks, _, _ = model.eval(image, diameter=diameter, flow_threshold=flow_threshold)
    sync_device()
    elapsed = time.perf_counter() - start
    return masks.astype(np.uint32), elapsed


def stitch_roi_masks(model, gray, rois, diameter, flow_threshold):
    """
    Run Cellpose on each ROI independently and stitch the instance labels back
    into a full-image mask with globally unique IDs.
    """
    full_masks = np.zeros(gray.shape, dtype=np.uint32)
    total_eval_time = 0.0
    next_label = 1

    for roi in rois:
        x1, y1, x2, y2 = roi["bbox"]
        crop = gray[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        roi_masks, eval_time = evaluate_cellpose(
            model, crop, diameter=diameter, flow_threshold=flow_threshold
        )
        total_eval_time += eval_time
        if roi_masks.max() == 0:
            continue
        roi_masks = roi_masks.astype(np.uint32)
        roi_masks[roi_masks > 0] += next_label - 1
        target = full_masks[y1:y2, x1:x2]
        target[roi_masks > 0] = roi_masks[roi_masks > 0]
        full_masks[y1:y2, x1:x2] = target
        next_label = int(full_masks.max()) + 1

    return full_masks, total_eval_time


def save_ablation_summary_figure(image_path, image_bgr, method_results, output_dir):
    stem = sanitize_stem(image_path)
    vis_dir = os.path.join(output_dir, "visuals")
    ensure_dir(vis_dir)

    n = len(method_results)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 9))
    if n == 1:
        axes = np.array(axes).reshape(2, 1)

    for col, result in enumerate(method_results):
        overlay = result["overlay"]
        mask_color = colorize_mask(result["masks"])

        ax0 = axes[0, col]
        ax0.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax0.set_title(
            f"{result['label']}\n"
            f"time={result['total_time_s']:.3f}s | inst={result['num_instances']}\n"
            f"d={result['diameter']} [{result['diameter_source']}]",
            fontsize=11,
        )
        ax0.axis("off")

        ax1 = axes[1, col]
        ax1.imshow(cv2.cvtColor(mask_color, cv2.COLOR_BGR2RGB))
        ax1.set_title(
            f"roi={result['roi_fraction'] * 100:.1f}% | rois={result['num_rois']}",
            fontsize=10,
        )
        ax1.axis("off")

    fig.suptitle(f"SeedPose Ablation Summary: {os.path.basename(image_path)}", fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(vis_dir, f"{stem}_ablation_summary.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def save_seed_crops(image_path, image_bgr, masks, method_name, output_dir, min_area=50):
    stem = sanitize_stem(image_path)
    save_dir = os.path.join(output_dir, "seeds", stem, method_name)
    ensure_dir(save_dir)

    kernel_ids = np.unique(masks)
    kernel_ids = kernel_ids[kernel_ids > 0]
    saved = 0

    for kernel_id in kernel_ids:
        mask = masks == kernel_id
        area = int(mask.sum())
        if area < min_area:
            continue
        ys, xs = np.where(mask)
        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()

        crop = image_bgr[y1 : y2 + 1, x1 : x2 + 1].copy()
        crop_mask = mask[y1 : y2 + 1, x1 : x2 + 1]
        crop[~crop_mask] = 0
        out_path = os.path.join(save_dir, f"seed_{int(kernel_id):04d}.png")
        cv2.imwrite(out_path, crop)
        saved += 1

    return save_dir, saved


def choose_best_fixed_diameter(image_summaries, baseline_instances):
    fixed_rows = [s for s in image_summaries if s["diameter_mode"] == "fixed"]
    if not fixed_rows:
        return None

    def score(row):
        instance_gap = abs(row["num_instances_last"] - baseline_instances)
        return (instance_gap, row["total_time_median_s"])

    return min(fixed_rows, key=score)


def save_full_debug(debug_root, image_bgr, gray, masks):
    save_image(os.path.join(debug_root, "full", "01_original.jpg"), image_bgr)
    save_image(os.path.join(debug_root, "full", "02_gray.jpg"), gray)
    save_image(os.path.join(debug_root, "full", "03_mask_color.jpg"), colorize_mask(masks))
    save_image(
        os.path.join(debug_root, "full", "04_overlay.jpg"),
        overlay_mask_on_image(image_bgr, masks),
    )


def save_roi_debug(debug_root, image_bgr, gray, masks, roi_debug, roi_region, merged_masks):
    save_image(os.path.join(debug_root, "roi", "01_original.jpg"), image_bgr)
    save_image(os.path.join(debug_root, "roi", "02_gray.jpg"), gray)
    save_image(os.path.join(debug_root, "roi", "03_blur.jpg"), roi_debug["blurred"])
    save_image(os.path.join(debug_root, "roi", "04_threshold.jpg"), roi_debug["binary"])
    save_image(os.path.join(debug_root, "roi", "05_open.jpg"), roi_debug["binary_open"])
    save_image(os.path.join(debug_root, "roi", "06_close.jpg"), roi_debug["binary_close"])
    save_image(
        os.path.join(debug_root, "roi", "07_candidate_rois.jpg"),
        draw_rois(image_bgr, roi_debug.get("candidate_rois", []), color=(0, 255, 255)),
    )
    save_image(
        os.path.join(debug_root, "roi", "08_merged_rois.jpg"),
        draw_rois(image_bgr, roi_debug.get("merged_rois", []), color=(0, 165, 255)),
    )
    save_image(os.path.join(debug_root, "roi", "09_roi_crop.jpg"), roi_region)
    save_image(os.path.join(debug_root, "roi", "10_roi_mask_color.jpg"), colorize_mask(merged_masks))
    save_image(os.path.join(debug_root, "roi", "11_full_mask_color.jpg"), colorize_mask(masks))
    save_image(
        os.path.join(debug_root, "roi", "12_overlay.jpg"),
        overlay_mask_on_image(image_bgr, masks),
    )


def benchmark_roi_method(
    model,
    image_bgr,
    gray,
    flow_threshold,
    roi_detector,
    diameter_mode,
    fixed_diameter=None,
    save_debug=False,
    debug_root=None,
):
    sync_device()
    prep_start = time.perf_counter()
    candidate_rois, roi_debug = build_candidate_rois(gray, detector=roi_detector)
    roi_diameter = estimate_diameter_from_rois(candidate_rois)
    diameter = fixed_diameter if diameter_mode == "fixed" and fixed_diameter is not None else roi_diameter
    if diameter is None:
        diameter, diameter_debug = estimate_kernel_diameter_from_image(gray)
    else:
        diameter_debug = None

    if candidate_rois:
        all_x1 = min(r["bbox"][0] for r in candidate_rois)
        all_y1 = min(r["bbox"][1] for r in candidate_rois)
        all_x2 = max(r["bbox"][2] for r in candidate_rois)
        all_y2 = max(r["bbox"][3] for r in candidate_rois)
        roi_region = gray[all_y1:all_y2, all_x1:all_x2]
        roi_pixels = sum(
            max(0, roi["bbox"][2] - roi["bbox"][0]) * max(0, roi["bbox"][3] - roi["bbox"][1])
            for roi in candidate_rois
        )
    else:
        roi_region = np.zeros((1, 1), dtype=gray.dtype)
        roi_pixels = 0
    sync_device()
    preprocess_time = time.perf_counter() - prep_start

    if candidate_rois:
        masks, eval_time = stitch_roi_masks(
            model, gray, candidate_rois, diameter=diameter, flow_threshold=flow_threshold
        )
        merged_masks = masks[all_y1:all_y2, all_x1:all_x2]
        roi_fraction = roi_pixels / float(gray.size)
    else:
        eval_time = 0.0
        masks = np.zeros(gray.shape, dtype=np.uint32)
        roi_fraction = 0.0
        merged_masks = np.zeros((1, 1), dtype=np.uint32)

    if save_debug and debug_root is not None:
        roi_bgr = cv2.cvtColor(roi_region, cv2.COLOR_GRAY2BGR)
        save_roi_debug(debug_root, image_bgr, gray, masks, roi_debug, roi_bgr, merged_masks)

    return {
        "method": "roi_preprocess",
        "roi_mode": "split",
        "roi_detector": roi_detector,
        "diameter_mode": diameter_mode,
        "preprocess_time_s": preprocess_time,
        "cellpose_time_s": eval_time,
        "total_time_s": preprocess_time + eval_time,
        "diameter": -1 if diameter is None else diameter,
        "diameter_source": (
            "user_fixed"
            if diameter_mode == "fixed" and fixed_diameter is not None
            else (
            "roi_area"
            if roi_diameter is not None
            else ("image_components" if diameter_debug is not None else "none")
            )
        ),
        "num_rois": len(candidate_rois),
        "roi_fraction": roi_fraction,
        "num_instances": int(masks.max()),
        "foreground_pixels": int(np.count_nonzero(masks)),
        "masks": masks,
        "overlay": overlay_mask_on_image(image_bgr, masks),
    }


def benchmark_full_image_method(
    model, image_bgr, gray, flow_threshold, diameter=None, save_debug=False, debug_root=None
):
    diameter_source = "user_fixed" if diameter is not None else "auto"
    if diameter is None:
        diameter, _ = estimate_kernel_diameter_from_image(gray)
        if diameter is not None:
            diameter_source = "image_components"
    masks, eval_time = evaluate_cellpose(
        model, gray, diameter=diameter, flow_threshold=flow_threshold
    )
    if save_debug and debug_root is not None:
        save_full_debug(debug_root, image_bgr, gray, masks)
    return {
        "method": "full_image",
        "roi_mode": "full_image",
        "roi_detector": "none",
        "diameter_mode": "fixed" if diameter_source == "user_fixed" else "auto",
        "preprocess_time_s": 0.0,
        "cellpose_time_s": eval_time,
        "total_time_s": eval_time,
        "diameter": -1 if diameter is None else diameter,
        "diameter_source": diameter_source,
        "num_rois": 1,
        "roi_fraction": 1.0,
        "num_instances": int(masks.max()),
        "foreground_pixels": int(np.count_nonzero(masks)),
        "masks": masks,
        "overlay": overlay_mask_on_image(image_bgr, masks),
    }


def summarize_runs(records, image_path, method_name):
    method_records = [r for r in records if r["image"] == image_path and r["method"] == method_name]
    return {
        "image": image_path,
        "method": method_name,
        "runs": len(method_records),
        "preprocess_time_mean_s": statistics.mean(r["preprocess_time_s"] for r in method_records),
        "cellpose_time_mean_s": statistics.mean(r["cellpose_time_s"] for r in method_records),
        "total_time_mean_s": statistics.mean(r["total_time_s"] for r in method_records),
        "total_time_median_s": statistics.median(r["total_time_s"] for r in method_records),
        "diameter_last": method_records[-1]["diameter"],
        "diameter_source_last": method_records[-1].get("diameter_source", ""),
        "roi_mode": method_records[-1].get("roi_mode", ""),
        "roi_detector": method_records[-1].get("roi_detector", ""),
        "diameter_mode": method_records[-1].get("diameter_mode", ""),
        "num_rois_last": method_records[-1]["num_rois"],
        "roi_fraction_last": method_records[-1]["roi_fraction"],
        "num_instances_last": method_records[-1]["num_instances"],
        "foreground_pixels_last": method_records[-1]["foreground_pixels"],
    }


def summarize_runs_by_name(records, image_path, ablation_name):
    method_records = [r for r in records if r["image"] == image_path and r["ablation"] == ablation_name]
    return {
        "image": image_path,
        "ablation": ablation_name,
        "label": method_records[-1]["label"],
        "runs": len(method_records),
        "roi_mode": method_records[-1]["roi_mode"],
        "roi_detector": method_records[-1]["roi_detector"],
        "diameter_mode": method_records[-1]["diameter_mode"],
        "preprocess_time_mean_s": statistics.mean(r["preprocess_time_s"] for r in method_records),
        "cellpose_time_mean_s": statistics.mean(r["cellpose_time_s"] for r in method_records),
        "total_time_mean_s": statistics.mean(r["total_time_s"] for r in method_records),
        "total_time_median_s": statistics.median(r["total_time_s"] for r in method_records),
        "diameter_last": method_records[-1]["diameter"],
        "diameter_source_last": method_records[-1].get("diameter_source", ""),
        "num_rois_last": method_records[-1]["num_rois"],
        "roi_fraction_last": method_records[-1]["roi_fraction"],
        "num_instances_last": method_records[-1]["num_instances"],
        "foreground_pixels_last": method_records[-1]["foreground_pixels"],
    }


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        filtered_rows = [
            {field: row.get(field, "") for field in fieldnames}
            for row in rows
        ]
        writer.writerows(filtered_rows)


def collect_image_paths(images=None, image_dir=None):
    image_paths = []
    if images:
        for item in images:
            matches = sorted(glob.glob(item))
            if matches:
                image_paths.extend(matches)
            else:
                image_paths.append(item)

    if image_dir:
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
            image_paths.extend(sorted(glob.glob(os.path.join(image_dir, pattern))))

    deduped = []
    seen = set()
    for path in image_paths:
        norm = os.path.normpath(path)
        if norm not in seen:
            seen.add(norm)
            deduped.append(norm)
    return deduped


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ROI-preprocessed Cellpose vs direct full-image Cellpose."
    )
    parser.add_argument("--images", nargs="+", default=DEFAULT_IMAGES, help="Image paths to benchmark.")
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Directory of images to benchmark. Supported extensions: jpg, jpeg, png, bmp, tif, tiff.",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Number of timed runs per method.")
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Untimed warmup runs per method before measurement.",
    )
    parser.add_argument(
        "--flow-threshold",
        type=float,
        default=0.4,
        help="Cellpose flow_threshold passed to both methods.",
    )
    parser.add_argument(
        "--fixed-diameter",
        type=float,
        default=None,
        help="Optional fixed diameter for ablation variants that use fixed diameter.",
    )
    parser.add_argument(
        "--diameter-sweep",
        nargs="+",
        type=float,
        default=None,
        help="Optional list of fixed diameters to test as A3 variants, e.g. --diameter-sweep 64 72 82 92",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV benchmark outputs.",
    )
    parser.add_argument(
        "--save-debug-images",
        action="store_true",
        help="Save per-image debug visualizations for both ROI and full-image methods.",
    )
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    print("=" * 72)
    print("Cellpose Speed Benchmark: SeedPose Ablation")
    print("=" * 72)
    print(f"Device: {'GPU' if use_gpu else 'CPU'}")
    print(f"Repeats: {args.repeats} | Warmup: {args.warmup}")
    print(f"Flow threshold: {args.flow_threshold}")
    print(f"Fixed diameter override: {args.fixed_diameter}")
    print(f"Diameter sweep: {args.diameter_sweep}")

    model = load_cellpose_model(use_gpu)
    ablation_methods = build_ablation_methods(args.diameter_sweep)
    image_paths = collect_image_paths(args.images, args.image_dir)

    if not image_paths:
        print("No input images found.")
        return

    print(f"Images to process: {len(image_paths)}")

    raw_rows = []
    summary_rows = []

    image_iter = tqdm(image_paths, desc="Images", unit="img") if len(image_paths) > 1 else image_paths
    for image_path in image_iter:
        image = cv2.imread(image_path)
        if image is None:
            print(f"\n[Skip] Cannot read image: {image_path}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        debug_root = os.path.join(args.output_dir, "debug", sanitize_stem(image_path))
        print(f"\nImage: {image_path}")
        print(f"Shape: {gray.shape[1]} x {gray.shape[0]}")

        image_estimated_diameter, _ = estimate_kernel_diameter_from_image(gray)
        fixed_diameter = (
            args.fixed_diameter if args.fixed_diameter is not None else image_estimated_diameter
        )
        print(f"Estimated image diameter: {image_estimated_diameter}")
        print(f"Default fixed diameter used by A3: {fixed_diameter}")

        for _ in range(args.warmup):
            for method_cfg in ablation_methods:
                if method_cfg["roi_mode"] == "full_image":
                    benchmark_full_image_method(model, image, gray, args.flow_threshold)
                else:
                    benchmark_roi_method(
                        model,
                        image,
                        gray,
                        args.flow_threshold,
                        roi_detector=method_cfg["roi_detector"],
                        diameter_mode=method_cfg["diameter_mode"],
                        fixed_diameter=(
                            method_cfg["fixed_diameter"]
                            if method_cfg["fixed_diameter"] is not None
                            else fixed_diameter
                        ),
                    )

        for run_idx in range(1, args.repeats + 1):
            save_debug = args.save_debug_images and run_idx == args.repeats
            run_results_for_figure = []
            for method_cfg in ablation_methods:
                method_debug_root = os.path.join(debug_root, method_cfg["name"])
                if method_cfg["roi_mode"] == "full_image":
                    stats = benchmark_full_image_method(
                        model,
                        image,
                        gray,
                        args.flow_threshold,
                        diameter=None,
                        save_debug=save_debug,
                        debug_root=method_debug_root,
                    )
                else:
                    stats = benchmark_roi_method(
                        model,
                        image,
                        gray,
                        args.flow_threshold,
                        roi_detector=method_cfg["roi_detector"],
                        diameter_mode=method_cfg["diameter_mode"],
                        fixed_diameter=(
                            method_cfg["fixed_diameter"]
                            if method_cfg["fixed_diameter"] is not None
                            else fixed_diameter
                        ),
                        save_debug=save_debug,
                        debug_root=method_debug_root,
                    )
                stats["image"] = image_path
                stats["run"] = run_idx
                stats["ablation"] = method_cfg["name"]
                stats["label"] = method_cfg["label"]
                if run_idx == args.repeats:
                    seed_dir, saved_count = save_seed_crops(
                        image_path,
                        image,
                        stats["masks"],
                        method_cfg["name"],
                        args.output_dir,
                    )
                    stats["seed_dir"] = seed_dir
                    stats["saved_seeds"] = saved_count
                raw_rows.append(stats)
                run_results_for_figure.append(stats)
                print(
                    f"  Run {run_idx} | {method_cfg['label']}: "
                    f"total={stats['total_time_s']:.3f}s "
                    f"(prep={stats['preprocess_time_s']:.3f}s, "
                    f"cellpose={stats['cellpose_time_s']:.3f}s, "
                    f"d={stats['diameter']} [{stats['diameter_source']}], "
                    f"rois={stats['num_rois']}, "
                    f"roi={stats['roi_fraction'] * 100:.1f}%, "
                    f"seeds={stats.get('saved_seeds', 0)})"
                )
            if run_idx == args.repeats:
                vis_path = save_ablation_summary_figure(
                    image_path, image, run_results_for_figure, args.output_dir
                )

        image_summaries = []
        for method_cfg in ablation_methods:
            summary = summarize_runs_by_name(raw_rows, image_path, method_cfg["name"])
            image_summaries.append(summary)
            summary_rows.append(summary)

        baseline_summary = next(s for s in image_summaries if s["ablation"] == "A0_full_image_auto")
        print("  Summary:")
        for summary in image_summaries:
            speedup = (
                baseline_summary["total_time_median_s"] / summary["total_time_median_s"]
                if summary["total_time_median_s"] > 0
                else float("inf")
            )
            print(
                f"    {summary['label']}: median={summary['total_time_median_s']:.3f}s, "
                f"speedup={speedup:.2f}x, "
                f"instances={summary['num_instances_last']}, "
                f"d={summary['diameter_last']} [{summary['diameter_source_last']}], "
                f"roi={summary['roi_fraction_last'] * 100:.1f}%"
            )
        best_fixed = choose_best_fixed_diameter(
            image_summaries, baseline_summary["num_instances_last"]
        )
        if best_fixed is not None:
            print(
                f"    Best fixed diameter: d={best_fixed['diameter_last']} "
                f"({best_fixed['label']}, median={best_fixed['total_time_median_s']:.3f}s, "
                f"instances={best_fixed['num_instances_last']})"
            )
        print(f"    Summary figure: {vis_path}")
        if args.save_debug_images:
            print(f"    Debug images: {debug_root}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    raw_csv = os.path.join(args.output_dir, f"cellpose_speed_raw_{timestamp}.csv")
    summary_csv = os.path.join(args.output_dir, f"cellpose_speed_summary_{timestamp}.csv")

    if raw_rows:
        raw_fields = [
            "image",
            "run",
            "method",
            "ablation",
            "label",
            "roi_mode",
            "roi_detector",
            "diameter_mode",
            "preprocess_time_s",
            "cellpose_time_s",
            "total_time_s",
            "diameter",
            "diameter_source",
            "num_rois",
            "roi_fraction",
            "num_instances",
            "foreground_pixels",
        ]
        summary_fields = [
            "image",
            "ablation",
            "label",
            "runs",
            "roi_mode",
            "roi_detector",
            "diameter_mode",
            "preprocess_time_mean_s",
            "cellpose_time_mean_s",
            "total_time_mean_s",
            "total_time_median_s",
            "diameter_last",
            "diameter_source_last",
            "num_rois_last",
            "roi_fraction_last",
            "num_instances_last",
            "foreground_pixels_last",
        ]
        write_csv(raw_csv, raw_rows, raw_fields)
        write_csv(summary_csv, summary_rows, summary_fields)
        print(f"\nRaw results: {raw_csv}")
        print(f"Summary: {summary_csv}")
    else:
        print("\nNo valid images were benchmarked.")


if __name__ == "__main__":
    main()
