#!/usr/bin/env bash
# run_all.sh — Pipeline completo Pregunta 3: detección de cumplimiento EPP con YOLO.
#
# Uso local/Colab:
#   export ROBOFLOW_API_KEY="tu_api_key"
#   bash run_all.sh
#
# Opciones útiles:
#   SKIP_DATASET_SETUP=1 bash run_all.sh        # usa data/ppe_yolo ya existente
#   FORCE_TRAIN=1 bash run_all.sh              # reentrena aunque existan checkpoints
#   ABLATION_CONFIG=configs/ablation_yolov8n_no_mosaic_full.yaml bash run_all.sh

set -euo pipefail

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

DATA_YAML="${DATA_YAML:-data/ppe_yolo/data.yaml}"
IMG_SIZE="${IMG_SIZE:-640}"

YOLOV8N_CONFIG="${YOLOV8N_CONFIG:-configs/yolov8n_baseline.yaml}"
YOLOV8S_CONFIG="${YOLOV8S_CONFIG:-configs/yolov8s_main.yaml}"
# Por defecto reproduce los resultados finales reportados: ablación rápida de 5 épocas.
# Para el protocolo completo de 30 épocas use:
# ABLATION_CONFIG=configs/ablation_yolov8n_no_mosaic_full.yaml bash run_all.sh
ABLATION_CONFIG="${ABLATION_CONFIG:-configs/ablation_yolov8n_no_mosaic.yaml}"

YOLOV8N_BEST="outputs/yolov8n_baseline/weights/best.pt"
YOLOV8S_BEST="outputs/yolov8s_main/weights/best.pt"
ABLATION_BEST="outputs/ablation_yolov8n_no_mosaic/weights/best.pt"

# Modelo usado para oclusión y demo. Se usa el baseline porque fue el mejor en los resultados finales.
ANALYSIS_MODEL="${ANALYSIS_MODEL:-${YOLOV8N_BEST}}"

mkdir -p outputs results logs

check_deps() {
python - <<'PYEOF'
import ultralytics
import roboflow
import cv2
import yaml
import numpy
import pandas
import sklearn
import gdown
PYEOF
}

train_if_needed() {
    local config="$1"
    local ckpt="$2"
    local label="$3"

    if [ -f "$ckpt" ] && [ "${FORCE_TRAIN:-0}" != "1" ]; then
        echo "✓ ${label}: checkpoint existente, se omite entrenamiento (${ckpt})"
    else
        echo "Entrenando ${label}..."
        python scripts/train.py --config "$config"
    fi

    if [ ! -f "$ckpt" ]; then
        echo "ERROR: No se encontró ${ckpt}"
        exit 1
    fi
}

echo "============================================================"
echo "  Pregunta 3 — Detección de cumplimiento EPP con YOLOv8"
echo "============================================================"
echo "PYTHONPATH=${PYTHONPATH}"
echo "DATA_YAML=${DATA_YAML}"
echo "ABLATION_CONFIG=${ABLATION_CONFIG}"
echo "ANALYSIS_MODEL=${ANALYSIS_MODEL}"
echo ""

echo "[0/9] Instalando/verificando dependencias..."
if check_deps >/dev/null 2>&1; then
    echo "✓ Dependencias listas"
else
    echo "Instalando dependencias desde requirements.txt..."
    pip install -r requirements.txt -q
    check_deps >/dev/null
    echo "✓ Dependencias instaladas y validadas"
fi

echo ""
echo "[1/9] Preparando datasets..."
if [ "${SKIP_DATASET_SETUP:-0}" = "1" ]; then
    echo "SKIP_DATASET_SETUP=1: se omite descarga/preparación."
else
    python scripts/setup_datasets.py --sources configs/data_sources.yaml
    python scripts/prepare_dataset.py \
        --sources configs/data_sources.yaml \
        --mapping configs/class_mapping.yaml \
        --out_dir data/ppe_yolo
fi

python scripts/validate_dataset.py --data "${DATA_YAML}"

echo ""
echo "[2/9] YOLOv8n baseline — 40 épocas, imgsz=640, mosaic ON..."
train_if_needed "${YOLOV8N_CONFIG}" "${YOLOV8N_BEST}" "YOLOv8n baseline"

echo ""
echo "[3/9] YOLOv8s principal — 60 épocas, imgsz=640, mosaic ON..."
train_if_needed "${YOLOV8S_CONFIG}" "${YOLOV8S_BEST}" "YOLOv8s principal"

echo ""
echo "[4/9] Ablación YOLOv8n sin mosaic..."
train_if_needed "${ABLATION_CONFIG}" "${ABLATION_BEST}" "Ablación YOLOv8n sin mosaic"

echo ""
echo "[5/9] Evaluación cuantitativa mAP@0.5 y mAP@0.5:0.95..."
python scripts/evaluate.py --model "${YOLOV8N_BEST}" --data "${DATA_YAML}" --out_dir results/metrics/yolov8n_baseline --imgsz "${IMG_SIZE}" --split test
python scripts/evaluate.py --model "${YOLOV8S_BEST}" --data "${DATA_YAML}" --out_dir results/metrics/yolov8s_main --imgsz "${IMG_SIZE}" --split test
python scripts/evaluate.py --model "${ABLATION_BEST}" --data "${DATA_YAML}" --out_dir results/metrics/ablation_yolov8n_no_mosaic --imgsz "${IMG_SIZE}" --split test

echo ""
echo "[6/9] Análisis de robustez ante oclusión/dificultad..."
python scripts/occlusion_analysis.py --model "${ANALYSIS_MODEL}" --data "${DATA_YAML}" --out_dir results/occlusion_analysis --imgsz "${IMG_SIZE}"

echo ""
echo "[7/9] Demo del verificador de cumplimiento basado en reglas..."
python scripts/compliance_demo.py --model "${ANALYSIS_MODEL}" --data "${DATA_YAML}" --out_dir results/compliance_demo --n_images 12 --conf 0.25

echo ""
echo "[8/9] Resumen final..."
python scripts/summarize_results.py | tee results/summary.txt

echo ""
echo "[9/9] Verificación de artefactos mínimos..."
python - <<'PYEOF'
from pathlib import Path
paths = [
    "data/ppe_yolo/data.yaml",
    "outputs/yolov8n_baseline/weights/best.pt",
    "outputs/yolov8s_main/weights/best.pt",
    "outputs/ablation_yolov8n_no_mosaic/weights/best.pt",
    "results/metrics/summary_metrics.csv",
    "results/metrics/yolov8n_baseline/metrics.json",
    "results/metrics/yolov8s_main/metrics.json",
    "results/metrics/ablation_yolov8n_no_mosaic/metrics.json",
    "results/occlusion_analysis/occlusion_summary.json",
    "results/compliance_demo/compliance_summary.json",
    "results/summary.txt",
]
missing = []
for p in paths:
    ok = Path(p).exists()
    print(f"{p:<75} {'OK' if ok else 'FALTA'}")
    if not ok:
        missing.append(p)
if missing:
    raise SystemExit("Artefactos faltantes: " + ", ".join(missing))
PYEOF

echo ""
echo "============================================================"
echo "  Pipeline completado"
echo "============================================================"
