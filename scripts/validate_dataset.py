#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ppe_yolo/data.yaml")
    args = parser.parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(f"No existe {data_yaml}")

    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    root = Path(data.get("path", data_yaml.parent)).resolve()
    names = data["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]

    print("Dataset:", data_yaml)
    print("Root:", root)
    print("Clases:", names)

    global_counts = Counter()

    for split in ["train", "val", "test"]:
        img_dir = root / data[split]
        lbl_dir = root / data[split].replace("images", "labels")
        imgs = [p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS] if img_dir.exists() else []
        labels = list(lbl_dir.rglob("*.txt")) if lbl_dir.exists() else []

        split_counts = Counter()
        for lbl in labels:
            for line in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cid = int(float(parts[0]))
                if 0 <= cid < len(names):
                    split_counts[names[cid]] += 1

        global_counts.update(split_counts)
        print(f"\n[{split}] imágenes={len(imgs)} labels={len(labels)} cajas={sum(split_counts.values())}")
        for cls in names:
            print(f"  {cls:<12}: {split_counts[cls]}")

        if split in {"train", "val"} and len(imgs) == 0:
            raise RuntimeError(f"El split {split} no tiene imágenes.")

    print("\nConteo global:")
    for cls in names:
        print(f"  {cls:<12}: {global_counts[cls]}")

    if global_counts["helmet"] == 0 or global_counts["no_helmet"] == 0:
        raise RuntimeError("El dataset final debe contener helmet y no_helmet.")

    print("\n✓ Validación finalizada correctamente")


if __name__ == "__main__":
    main()
