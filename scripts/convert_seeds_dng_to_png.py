from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "discussion_dataset" / "seeds"
OUT_ROOT = ROOT / "data" / "seeds_png"
VALID_EXTS = {".dng", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def convert_image(src_path: Path, dst_path: Path) -> dict:
    arr = iio.imread(src_path)
    img = Image.fromarray(arr)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst_path)
    return {
        "src": str(src_path),
        "dst": str(dst_path),
        "size": list(img.size),
        "mode": img.mode,
    }


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    converted = []

    for src_path in sorted(SRC_ROOT.rglob("*")):
        if not src_path.is_file():
            continue
        if src_path.suffix.lower() not in VALID_EXTS:
            continue
        rel = src_path.relative_to(SRC_ROOT)
        dst_path = OUT_ROOT / rel.with_suffix(".png")
        converted.append(convert_image(src_path, dst_path))
        print(f"Converted: {rel} -> {dst_path.relative_to(OUT_ROOT)}")

    summary = {
        "source_root": str(SRC_ROOT),
        "output_root": str(OUT_ROOT),
        "count": len(converted),
        "files": converted,
    }
    (OUT_ROOT / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved {len(converted)} converted files to {OUT_ROOT}")


if __name__ == "__main__":
    main()
