#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path


def fmt(x):
    try:
        x = float(x)
        if math.isnan(x):
            return "n/a"
        return f"{x:.4f}"
    except Exception:
        return "n/a"


def load(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    experiments = [
        ("YOLOv8n baseline", "results/metrics/yolov8n_baseline/metrics.json"),
        ("YOLOv8s principal", "results/metrics/yolov8s_main/metrics.json"),
        ("YOLOv8n no mosaic", "results/metrics/ablation_yolov8n_no_mosaic/metrics.json"),
    ]

    summary_rows = []

    print("\n================ RESULTADOS FINALES ================")
    print(f"{'Experimento':<28} {'mAP50':>8} {'mAP50-95':>10} {'Prec':>8} {'Recall':>8}")
    print("-" * 70)
    for name, path in experiments:
        m = load(path)
        if not m:
            print(f"{name:<28} {'FALTA':>8}")
            continue

        row = {
            "experimento": name,
            "map50": m.get("map50"),
            "map50_95": m.get("map50_95"),
            "map75": m.get("map75"),
            "precision_mean": m.get("precision_mean"),
            "recall_mean": m.get("recall_mean"),
        }
        summary_rows.append(row)
        print(
            f"{name:<28} {fmt(row['map50']):>8} {fmt(row['map50_95']):>10} "
            f"{fmt(row['precision_mean']):>8} {fmt(row['recall_mean']):>8}"
        )

    if summary_rows:
        write_csv(
            Path("results/metrics/summary_metrics.csv"),
            ["experimento", "map50", "map50_95", "map75", "precision_mean", "recall_mean"],
            summary_rows,
        )
        print("\n✓ Resumen comparativo guardado en results/metrics/summary_metrics.csv")

    occ = load("results/occlusion_analysis/occlusion_summary.json")
    if occ:
        occ_rows = []
        print("\nRobustez por oclusión/dificultad:")
        print(f"{'Grupo':<10} {'N':>6} {'mAP50':>8} {'mAP50-95':>10}")
        print("-" * 45)
        for k in ["baja", "media", "alta"]:
            if k in occ:
                row = {
                    "grupo": k,
                    "n_images": occ[k].get("n_images", 0),
                    "map50": occ[k].get("map50"),
                    "map50_95": occ[k].get("map50_95"),
                    "precision_mean": occ[k].get("precision_mean"),
                    "recall_mean": occ[k].get("recall_mean"),
                }
                occ_rows.append(row)
                print(f"{k:<10} {row['n_images']:>6} {fmt(row['map50']):>8} {fmt(row['map50_95']):>10}")

        if occ_rows:
            write_csv(
                Path("results/occlusion_analysis/occlusion_summary.csv"),
                ["grupo", "n_images", "map50", "map50_95", "precision_mean", "recall_mean"],
                occ_rows,
            )
            print("✓ Resumen de oclusión guardado en results/occlusion_analysis/occlusion_summary.csv")

    comp = load("results/compliance_demo/compliance_summary.json")
    if comp:
        total = sum(x.get("num_workers", 0) for x in comp)
        ok = sum(x.get("num_compliant", 0) for x in comp)
        rate = ok / total if total else 0.0
        print("\nDemo de cumplimiento:")
        print(f"Frames: {len(comp)} | Trabajadores/pseudo-trabajadores: {total} | Cumplimiento promedio: {rate:.4f}")

        write_csv(
            Path("results/compliance_demo/compliance_summary.csv"),
            ["frames", "workers", "compliant", "non_compliant", "compliance_rate"],
            [{
                "frames": len(comp),
                "workers": total,
                "compliant": ok,
                "non_compliant": total - ok,
                "compliance_rate": rate,
            }],
        )
        print("✓ Resumen de cumplimiento guardado en results/compliance_demo/compliance_summary.csv")

    print("\nResultados detallados en results/ y checkpoints en outputs/.")


if __name__ == "__main__":
    main()
