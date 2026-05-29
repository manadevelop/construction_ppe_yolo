#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO


def resolve_output_dir(cfg: dict[str, Any]) -> Path:
    """
    Resuelve de forma robusta la carpeta final del experimento.

    Casos soportados:
    1. output_dir: outputs/yolov8n_baseline
    2. project: outputs
       name: yolov8n_baseline
    3. project: runs/detect
       name: outputs/yolov8n_baseline

    El caso 3 se corrige automáticamente para evitar que los pesos terminen en:
    runs/detect/outputs/yolov8n_baseline

    La salida esperada del proyecto siempre debe quedar en:
    outputs/<nombre_experimento>
    """

    if cfg.get("output_dir"):
        out_dir = Path(str(cfg["output_dir"]))
        return out_dir

    name_value = str(cfg.get("name", cfg.get("experiment_name", "experiment")))
    name_path = Path(name_value)

    # Si name viene como ruta, por ejemplo outputs/yolov8n_baseline,
    # se usa directamente esa ruta y se ignora project.
    if name_path.parent != Path("."):
        return name_path

    project_value = str(cfg.get("project", "outputs"))
    project_path = Path(project_value)

    return project_path / name_path


def copy_legacy_run_if_needed(expected_out_dir: Path) -> None:
    """
    Recupera resultados si, por alguna configuración previa, el entrenamiento
    quedó guardado en runs/detect/outputs/<experimento>.

    Esto evita perder entrenamientos ya ejecutados.
    """
    legacy_dir = Path("runs") / "detect" / "outputs" / expected_out_dir.name

    expected_best = expected_out_dir / "weights" / "best.pt"
    legacy_best = legacy_dir / "weights" / "best.pt"

    if expected_best.exists():
        return

    if legacy_best.exists():
        expected_out_dir.parent.mkdir(parents=True, exist_ok=True)

        if expected_out_dir.exists():
            shutil.rmtree(expected_out_dir)

        shutil.copytree(legacy_dir, expected_out_dir)
        print(f"✓ Resultados recuperados desde {legacy_dir} hacia {expected_out_dir}")


def save_training_config(out_dir: Path, config_path: str, cfg: dict[str, Any], train_args: dict[str, Any]) -> None:
    """
    Guarda una copia de la configuración usada para trazabilidad.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "config_file": config_path,
        "config": cfg,
        "train_args": train_args,
    }

    with open(out_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)

    if not config_path.exists():
        raise FileNotFoundError(f"No existe el archivo de configuración: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise RuntimeError(f"Configuración inválida en {config_path}")

    if "model" not in cfg:
        raise RuntimeError(f"Falta 'model' en {config_path}")

    if "data" not in cfg:
        raise RuntimeError(f"Falta 'data' en {config_path}")

    out_dir = resolve_output_dir(cfg).resolve()

    # Separar correctamente project y name para que la salida quede exactamente en:
    # outputs/<experimento>
    project_dir = out_dir.parent
    experiment_name = out_dir.name

    model = YOLO(cfg["model"])

    device = cfg.get("device", "auto")
    if device == "auto":
        device = None

    train_args: dict[str, Any] = {
        "data": cfg["data"],
        "epochs": int(cfg.get("epochs", 50)),
        "imgsz": int(cfg.get("imgsz", 640)),
        "batch": int(cfg.get("batch", 16)),
        "patience": int(cfg.get("patience", 15)),
        "project": str(project_dir),
        "name": experiment_name,
        "workers": int(cfg.get("workers", 2)),
        "seed": int(cfg.get("seed", 42)),
        "optimizer": cfg.get("optimizer", "auto"),
        "mosaic": float(cfg.get("mosaic", 1.0)),
        "close_mosaic": int(cfg.get("close_mosaic", 10)),
        "cache": bool(cfg.get("cache", False)),
        "plots": bool(cfg.get("plots", True)),
        "exist_ok": True,
    }

    # Parámetros opcionales frecuentes.
    optional_keys = [
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        "warmup_epochs",
        "warmup_momentum",
        "warmup_bias_lr",
        "box",
        "cls",
        "dfl",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "flipud",
        "fliplr",
        "mixup",
        "copy_paste",
        "cos_lr",
        "amp",
        "save_period",
    ]

    for key in optional_keys:
        if key in cfg:
            train_args[key] = cfg[key]

    if device is not None:
        train_args["device"] = device

    print("Entrenando con configuración:")
    print(json.dumps(train_args, indent=2, ensure_ascii=False))
    print(f"\nCarpeta de salida esperada: {out_dir}")
    print(f"Checkpoint esperado: {out_dir / 'weights' / 'best.pt'}\n")

    model.train(**train_args)

    # Si por alguna razón existiera una salida heredada, se intenta recuperar.
    copy_legacy_run_if_needed(out_dir)

    best_ckpt = out_dir / "weights" / "best.pt"
    last_ckpt = out_dir / "weights" / "last.pt"

    if not best_ckpt.exists():
        # Diagnóstico útil antes de fallar.
        possible = sorted(Path(".").glob("**/weights/best.pt"))
        print("\nNo se encontró el checkpoint esperado.")
        print(f"Esperado: {best_ckpt}")
        print("\nCheckpoints encontrados:")
        for p in possible[:30]:
            print(f"  - {p}")

        raise FileNotFoundError(f"No se encontró {best_ckpt}")

    save_training_config(
        out_dir=out_dir,
        config_path=str(config_path),
        cfg=cfg,
        train_args=train_args,
    )

    print("\n✓ Entrenamiento finalizado correctamente")
    print(f"✓ Carpeta de salida: {out_dir}")
    print(f"✓ Best checkpoint: {best_ckpt}")

    if last_ckpt.exists():
        print(f"✓ Last checkpoint: {last_ckpt}")


if __name__ == "__main__":
    main()