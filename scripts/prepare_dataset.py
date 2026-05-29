#!/usr/bin/env python3
"""Convierte y fusiona las tres fuentes en un dataset YOLOv8 único."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.convert_shwd import convert_shwd_to_yolo
from data.merge_yolo import merge_datasets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default="configs/data_sources.yaml")
    parser.add_argument("--mapping", default="configs/class_mapping.yaml")
    parser.add_argument("--out_dir", default="data/ppe_yolo")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.sources, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    roots = []

    for name, src in cfg.get("roboflow", {}).items():
        if src.get("enabled", True):
            roots.append(src["out_dir"])

    shwd = cfg.get("shwd", {})
    if shwd.get("enabled", True):
        converted = Path(shwd["converted_dir"])
        if not (converted / "data.yaml").exists():
            convert_shwd_to_yolo(shwd["out_dir"], converted)
        roots.append(converted)

    if not roots:
        raise RuntimeError("No hay fuentes habilitadas para fusionar.")

    merge_datasets(roots, args.out_dir, args.mapping, seed=args.seed)


if __name__ == "__main__":
    main()
