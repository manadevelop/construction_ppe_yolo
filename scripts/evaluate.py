#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml
from ultralytics import YOLO


def safe_float(x):
    try:
        x = float(x)
        return None if math.isnan(x) else x
    except Exception:
        return None


def evaluate(model_path: str, data_yaml: str, out_dir: str, imgsz: int = 640, split: str = "test"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)
    metrics = model.val(
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        project=str(out_dir),
        name="val",
        exist_ok=True,
        plots=True,
    )

    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]

    result = {
        "model_path": model_path,
        "data": data_yaml,
        "split": split,
        "imgsz": imgsz,
        "map50": safe_float(getattr(metrics.box, "map50", None)),
        "map50_95": safe_float(getattr(metrics.box, "map", None)),
        "map75": safe_float(getattr(metrics.box, "map75", None)),
        "precision_mean": safe_float(getattr(metrics.box, "mp", None)),
        "recall_mean": safe_float(getattr(metrics.box, "mr", None)),
        "per_class": {},
    }

    maps = getattr(metrics.box, "maps", None)
    if maps is not None:
        for i, name in enumerate(names):
            if i < len(maps):
                result["per_class"][name] = {"map50_95": safe_float(maps[i])}

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n=== MÉTRICAS ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n✓ Métricas guardadas en {out_dir / 'metrics.json'}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="data/ppe_yolo/data.yaml")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()
    evaluate(args.model, args.data, args.out_dir, args.imgsz, args.split)


if __name__ == "__main__":
    main()
