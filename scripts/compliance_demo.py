#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compliance.rules import evaluate_compliance

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_names(data_yaml: str | Path):
    with open(data_yaml, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    names = d["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    root = Path(d.get("path", Path(data_yaml).parent)).resolve()
    return names, root, d


def detections_from_result(result, names):
    dets = []
    boxes = result.boxes
    if boxes is None:
        return dets
    for box in boxes:
        cls_id = int(box.cls.detach().cpu().item())
        conf = float(box.conf.detach().cpu().item())
        xyxy = box.xyxy.detach().cpu().numpy().reshape(-1).tolist()
        if 0 <= cls_id < len(names):
            dets.append({"class_id": cls_id, "class_name": names[cls_id], "confidence": conf, "xyxy": xyxy})
    return dets


def draw_compliance(image, compliance):
    for w in compliance["workers"]:
        x1, y1, x2, y2 = map(int, w["worker_box"])
        status = w["status"]
        color = (0, 200, 0) if status == "compliant" else (0, 0, 255)
        label = "CUMPLE" if status == "compliant" else "NO CUMPLE: " + ",".join(w["violation_reasons"])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    text = f"Cumplimiento: {compliance['num_compliant']}/{compliance['num_workers']} = {compliance['compliance_rate']:.2f}"
    cv2.putText(image, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
    cv2.putText(image, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="data/ppe_yolo/data.yaml")
    parser.add_argument("--source", default=None, help="Directorio de imágenes. Si no se indica, usa test.")
    parser.add_argument("--out_dir", default="results/compliance_demo")
    parser.add_argument("--n_images", type=int, default=12)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--require_vest", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names, root, data = load_names(args.data)
    if args.source:
        img_dir = Path(args.source)
    else:
        img_dir = root / data.get("test", "images/test")

    imgs = [p for p in sorted(img_dir.rglob("*")) if p.suffix.lower() in IMG_EXTS][: args.n_images]
    if not imgs:
        raise RuntimeError(f"No se encontraron imágenes en {img_dir}")

    model = YOLO(args.model)
    summary = []

    for idx, img_path in enumerate(imgs):
        result = model.predict(str(img_path), conf=args.conf, verbose=False)[0]
        dets = detections_from_result(result, names)
        comp = evaluate_compliance(dets, require_vest=args.require_vest)
        comp["frame_id"] = idx
        comp["image"] = str(img_path)
        summary.append(comp)

        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image = draw_compliance(image, comp)
        cv2.imwrite(str(out_dir / f"frame_{idx:03d}.jpg"), image)

    with open(out_dir / "compliance_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"✓ Demo de cumplimiento guardado en {out_dir}")


if __name__ == "__main__":
    main()
