#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

from evaluate import evaluate

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_labels(lbl_path: Path):
    rows = []
    if not lbl_path.exists():
        return rows
    for line in lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, x, y, w, h = int(float(parts[0])), *map(float, parts[1:5])
        rows.append((cls, x, y, w, h))
    return rows


def overlap_score(rows):
    # Heurística simple: muchos objetos con cajas cercanas aumentan dificultad.
    return min(1.0, max(0, len(rows) - 1) / 6.0)


def image_difficulty(rows):
    if not rows:
        return "baja"
    areas = [w * h for _, _, _, w, h in rows]
    min_area = min(areas)
    small_score = 1.0 if min_area < 0.005 else (0.5 if min_area < 0.02 else 0.0)
    ov = overlap_score(rows)
    score = 0.65 * small_score + 0.35 * ov
    if score < 0.30:
        return "baja"
    if score < 0.65:
        return "media"
    return "alta"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="data/ppe_yolo/data.yaml")
    parser.add_argument("--out_dir", default="results/occlusion_analysis")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.data, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    root = Path(data.get("path", Path(args.data).parent)).resolve()
    names = data["names"]

    img_dir = root / data.get("test", "images/test")
    lbl_dir = root / data.get("test", "images/test").replace("images", "labels")

    groups = {"baja": [], "media": [], "alta": []}
    for img in sorted(img_dir.rglob("*")):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        lbl = lbl_dir / f"{img.stem}.txt"
        rows = read_labels(lbl)
        group = image_difficulty(rows)
        groups[group].append((img, lbl))

    summary = {}
    for group, pairs in groups.items():
        group_root = out_dir / f"dataset_{group}"
        (group_root / "images" / "test").mkdir(parents=True, exist_ok=True)
        (group_root / "labels" / "test").mkdir(parents=True, exist_ok=True)
        # train/val dummy apuntan a test para que Ultralytics acepte YAML.
        (group_root / "images" / "train").mkdir(parents=True, exist_ok=True)
        (group_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (group_root / "images" / "val").mkdir(parents=True, exist_ok=True)
        (group_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

        for img, lbl in pairs:
            shutil.copy2(img, group_root / "images" / "test" / img.name)
            if lbl.exists():
                shutil.copy2(lbl, group_root / "labels" / "test" / lbl.name)

        group_yaml = group_root / "data.yaml"
        with open(group_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump({
                "path": str(group_root.resolve()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": names,
            }, f, sort_keys=False, allow_unicode=True)

        summary[group] = {"n_images": len(pairs)}
        if pairs:
            metrics = evaluate(args.model, str(group_yaml), str(out_dir / f"eval_{group}"), imgsz=args.imgsz, split="test")
            summary[group].update(metrics)

    with open(out_dir / "occlusion_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nResumen de oclusión / dificultad:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
