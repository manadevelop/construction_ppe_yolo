# Detección de Cumplimiento de Normas de Seguridad en Obras de Construcción (YOLO)

Proyecto desarrollado para la **Pregunta 3** del EP de Redes Neuronales y Aprendizaje Profundo.

**Curso:** Redes Neuronales y Aprendizaje Profundo  
**Docente:** Ph.D. Aldo Camargo  
**Universidad:** Universidad Nacional de Ingeniería — Maestría en Inteligencia Artificial  

**Integrantes:**

- Victor Fernando Montes Jaramillo — victor.montes.j@uni.pe
- Alex Celestino León Pacheco — alex.leon.p@uni.pe
- Edwin Jhon Minchán Ramos — edwin.minchan.r@uni.pe
- Marco Antonio Nina Aguilar — marco.nina.a@uni.pe

El objetivo es entrenar y evaluar modelos YOLOv8 para detectar trabajadores y elementos de Equipo de Protección Personal (EPP), principalmente **casco** y **chaleco**, y construir un verificador de cumplimiento basado en reglas que marque trabajadores conformes y no conformes por fotograma.

---

## 1. Fuentes de datos

Se usan las tres fuentes solicitadas por el enunciado:

1. **Safety Helmet Detection Dataset** — Roboflow Universe  
   URL: <https://universe.roboflow.com/safety-helmet-kfkub/safety-helmet-ifjeb>  
   Rol: fuente principal de entrenamiento.

2. **PPE Detection Dataset** — Roboflow Universe  
   URL: <https://universe.roboflow.com/testcasque/ppe-detection-qlq3d>  
   Rol: fuente complementaria para ampliar variabilidad y clases de EPP.

3. **SHWD — Safety Helmet Wearing Dataset**  
   Dataset original: <https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset>  
   En este proyecto se usa una copia real en formato **Pascal VOC/XML** descargada desde Google Drive mediante `gdown`, configurada en `configs/data_sources.yaml`.  
   Rol: refuerzo de ejemplos negativos explícitos de `no_helmet`.

Los datasets de Roboflow requieren `ROBOFLOW_API_KEY`. SHWD se descarga desde Google Drive y luego se convierte automáticamente de Pascal VOC/XML a YOLO mediante `src/data/convert_shwd.py`.

---

## 2. Taxonomía final

Todas las fuentes se fusionan en una taxonomía común:

```yaml
0: person
1: helmet
2: no_helmet
3: vest
4: no_vest
```

Clases fuera del alcance directo de la pregunta, como `boots`, `gloves` o `goggles`, se ignoran durante la fusión del dataset. El mapeo se define en:

```text
configs/class_mapping.yaml
```

---

## 3. Estructura del repositorio

```text
construction_ppe_yolo/
├── README.md
├── requirements.txt
├── run_all.sh
├── colab_runner.ipynb
├── .gitignore
├── configs/
│   ├── data_sources.yaml
│   ├── class_mapping.yaml
│   ├── yolov8n_baseline.yaml
│   ├── yolov8s_main.yaml
│   ├── ablation_yolov8n_no_mosaic.yaml
│   └── ablation_yolov8n_no_mosaic_full.yaml
├── scripts/
│   ├── setup_datasets.py
│   ├── prepare_dataset.py
│   ├── validate_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── occlusion_analysis.py
│   ├── compliance_demo.py
│   └── summarize_results.py
├── src/
│   ├── compliance/
│   │   ├── associate.py
│   │   └── rules.py
│   ├── data/
│   │   ├── convert_shwd.py
│   │   └── merge_yolo.py
│   └── utils/
│       └── logger.py
├── data/       # generado localmente, ignorado por Git
├── outputs/    # checkpoints y salidas de entrenamiento, ignorado por Git
├── results/    # métricas y figuras finales, ignorado por Git
└── reports/
    └── main.tex
```

---

## 4. Instalación

```bash
pip install -r requirements.txt
```

Dependencias principales:

```text
ultralytics
roboflow
gdown
opencv-python
numpy
pandas
matplotlib
scikit-learn
PyYAML
Pillow
lxml
```

---

## 5. Configurar Roboflow API Key

En local:

```bash
export ROBOFLOW_API_KEY="TU_PRIVATE_API_KEY"
```

En Google Colab:

```python
import os, getpass
os.environ["ROBOFLOW_API_KEY"] = getpass.getpass("ROBOFLOW_API_KEY: ")
```

No subir la API Key a GitHub.

---

## 6. Ejecución completa con un solo comando

El requisito principal de reproducibilidad se cumple con:

```bash
bash run_all.sh
```

El pipeline ejecuta:

1. Verificación/instalación de dependencias.
2. Descarga de Safety Helmet y PPE Detection desde Roboflow.
3. Descarga de SHWD desde Google Drive.
4. Conversión SHWD Pascal VOC/XML → YOLO.
5. Fusión de las tres fuentes en `data/ppe_yolo`.
6. Validación del dataset final.
7. Entrenamiento de YOLOv8n baseline.
8. Entrenamiento de YOLOv8s principal.
9. Ablación YOLOv8n sin mosaic.
10. Evaluación cuantitativa en test.
11. Análisis de robustez por oclusión/dificultad.
12. Demo del verificador de cumplimiento.
13. Resumen final y verificación de artefactos.

Opciones útiles:

```bash
# Usar dataset ya preparado y evitar descargar/fusionar otra vez
SKIP_DATASET_SETUP=1 bash run_all.sh

# Forzar reentrenamiento aunque existan checkpoints
FORCE_TRAIN=1 bash run_all.sh

# Ejecutar la ablación completa de 30 épocas
ABLATION_CONFIG=configs/ablation_yolov8n_no_mosaic_full.yaml bash run_all.sh
```

---

## 7. Experimentos configurados

| Experimento | Modelo | Épocas | ImgSz | Mosaic | Salida |
|---|---:|---:|---:|---:|---|
| Baseline | YOLOv8n | 40 | 640 | Sí | `outputs/yolov8n_baseline/` |
| Principal | YOLOv8s | 60 | 640 | Sí | `outputs/yolov8s_main/` |
| Ablación reportada | YOLOv8n | 5 | 640 | No | `outputs/ablation_yolov8n_no_mosaic/` |
| Ablación completa opcional | YOLOv8n | 30 | 640 | No | `outputs/ablation_yolov8n_no_mosaic/` |

La ablación reportada se ejecutó con 5 épocas por restricción de recursos en Colab. El repositorio conserva una configuración opcional de 30 épocas para repetir el protocolo completo.

Checkpoints esperados:

```text
outputs/yolov8n_baseline/weights/best.pt
outputs/yolov8s_main/weights/best.pt
outputs/ablation_yolov8n_no_mosaic/weights/best.pt
```

---

## 8. Resultados finales obtenidos

Evaluación sobre el split `test` del dataset unificado:

| Experimento | mAP@0.5 | mAP@0.5:0.95 | Precisión | Recall |
|---|---:|---:|---:|---:|
| YOLOv8n baseline | 0.6859 | 0.4299 | 0.8926 | 0.6558 |
| YOLOv8s principal | 0.5548 | 0.3142 | 0.7423 | 0.5457 |
| YOLOv8n sin mosaic | 0.6429 | 0.3879 | 0.8608 | 0.5781 |

Archivos generados:

```text
results/metrics/summary_metrics.csv
results/metrics/yolov8n_baseline/metrics.json
results/metrics/yolov8s_main/metrics.json
results/metrics/ablation_yolov8n_no_mosaic/metrics.json
```

---

## 9. Resultados por clase

El archivo `metrics.json` de cada experimento incluye `per_class` con mAP@0.5:0.95 por clase. En el mejor experimento global, YOLOv8n baseline, se obtuvo:

| Clase | mAP@0.5:0.95 |
|---|---:|
| person | 0.0047 |
| helmet | 0.6149 |
| no_helmet | 0.4712 |
| vest | 0.7393 |
| no_vest | 0.3196 |

La clase `person` tiene bajo desempeño porque las fuentes usadas están más orientadas a casco/cabeza/EPP que a cuerpo completo. Esta limitación debe discutirse en el informe.

---

## 10. Análisis de oclusión/dificultad

`scripts/occlusion_analysis.py` particiona el test set en tres grupos mediante una heurística semiautomática basada en tamaño relativo de cajas y número de objetos por imagen:

- baja dificultad/oclusión;
- media dificultad/oclusión;
- alta dificultad/oclusión.

Resultados obtenidos con el modelo YOLOv8n baseline:

| Grupo | N imágenes | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|
| Baja | 475 | 0.8204 | 0.6047 |
| Media | 420 | 0.7241 | 0.5256 |
| Alta | 839 | 0.6751 | 0.3762 |

La caída de mAP@0.5:0.95 entre baja y alta dificultad es aproximadamente 0.2285, consistente con la degradación esperada ante oclusión, objetos pequeños y escenas más saturadas.

---

## 11. Verificador de cumplimiento basado en reglas

El módulo `src/compliance/rules.py` recibe detecciones por imagen y aplica reglas:

- Si hay cajas `person`, se asocian `helmet/no_helmet` con la región superior del trabajador.
- Se asocian `vest/no_vest` con la región del torso.
- Si falta casco o chaleco requerido, el trabajador se marca como `non_compliant`.
- Si no hay cajas `person`, se usa una lógica fallback con detecciones `helmet/no_helmet` como pseudo-trabajadores.

Ejemplo de ejecución:

```bash
python scripts/compliance_demo.py \
  --model outputs/yolov8n_baseline/weights/best.pt \
  --data data/ppe_yolo/data.yaml \
  --out_dir results/compliance_demo \
  --n_images 12 \
  --conf 0.25
```

Resultado obtenido:

```text
Frames: 12
Trabajadores/pseudo-trabajadores: 18
Cumplimiento promedio: 0.3333
```

---

## 12. Ejecución por partes

```bash
# Descargar fuentes
python scripts/setup_datasets.py --sources configs/data_sources.yaml

# Preparar dataset final
python scripts/prepare_dataset.py \
  --sources configs/data_sources.yaml \
  --mapping configs/class_mapping.yaml \
  --out_dir data/ppe_yolo

# Validar dataset
python scripts/validate_dataset.py --data data/ppe_yolo/data.yaml

# Entrenar baseline
python scripts/train.py --config configs/yolov8n_baseline.yaml

# Entrenar modelo principal
python scripts/train.py --config configs/yolov8s_main.yaml

# Entrenar ablación reportada
python scripts/train.py --config configs/ablation_yolov8n_no_mosaic.yaml

# Evaluar un modelo
python scripts/evaluate.py \
  --model outputs/yolov8n_baseline/weights/best.pt \
  --data data/ppe_yolo/data.yaml \
  --out_dir results/metrics/yolov8n_baseline \
  --imgsz 640 \
  --split test

# Análisis de oclusión
python scripts/occlusion_analysis.py \
  --model outputs/yolov8n_baseline/weights/best.pt \
  --data data/ppe_yolo/data.yaml \
  --out_dir results/occlusion_analysis \
  --imgsz 640

# Demo de cumplimiento
python scripts/compliance_demo.py \
  --model outputs/yolov8n_baseline/weights/best.pt \
  --data data/ppe_yolo/data.yaml \
  --out_dir results/compliance_demo \
  --n_images 12 \
  --conf 0.25

# Resumen final
python scripts/summarize_results.py | tee results/summary.txt
```

---

## 13. Google Colab

El proyecto incluye `colab_runner.ipynb`. Flujo recomendado:

1. Abrir el notebook en Google Colab.
2. Activar GPU: `Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU`.
3. Ingresar `ROBOFLOW_API_KEY` cuando se solicite.
4. Ejecutar las celdas en orden.
5. Revisar `outputs/` y `results/`.
6. Respaldar resultados en Google Drive.

La carpeta sugerida de respaldo es:

```text
/content/drive/MyDrive/EP01/pregunta3/resultados/
```

---

## 14. Consideraciones éticas

Este sistema debe considerarse una herramienta de apoyo a inspección preventiva, no un mecanismo automático de sanción. Los falsos negativos pueden dejar pasar situaciones de riesgo real; los falsos positivos pueden marcar injustamente a un trabajador que sí cumple. Además, el uso de cámaras en obras introduce preocupaciones de privacidad, vigilancia laboral, consentimiento, sesgo por tipo de uniforme, iluminación, país, cámara y ángulo de captura. Cualquier despliegue real debe incluir revisión humana, políticas claras de uso, minimización de datos e indicadores de incertidumbre.

---

## 15. Referencias principales

1. Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. You Only Look Once: Unified, Real-Time Object Detection.
2. Jocher, G. et al. YOLO object detection software and documentation.
3. Roboflow Universe documentation.
4. Safety Helmet Detection Dataset, Roboflow Universe.
5. PPE Detection Dataset, Roboflow Universe.
6. NJVisionPower. Safety Helmet Wearing Dataset.
