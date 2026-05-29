#!/usr/bin/env python3
"""
Descarga las fuentes de datos de la Pregunta 3.

Fuentes:
- Safety Helmet: Roboflow API.
- PPE Detection: Roboflow API.
- SHWD: Google Drive, dataset real en formato Pascal VOC.

Nota:
SHWD no se obtiene mediante git clone porque el repositorio público no incluye
directamente todos los XML e imágenes del dataset. El dataset real se descarga
como archivo externo y luego se convierte en prepare_dataset.py.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import yaml
from roboflow import Roboflow


def has_pascal_voc_xml(root: Path) -> bool:
    """Verifica si una carpeta contiene anotaciones Pascal VOC."""
    return root.exists() and any(root.rglob("*.xml"))


def download_roboflow_source(
    api_key: str,
    name: str,
    cfg: dict,
    force: bool = False,
) -> None:
    """Descarga una fuente Roboflow en formato YOLO."""
    out_dir = Path(cfg["out_dir"])
    marker = out_dir / ".download_complete"

    if marker.exists() and not force:
        print(f"✓ {name}: ya descargado en {out_dir}")
        return

    if out_dir.exists() and force:
        shutil.rmtree(out_dir)

    out_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"Descargando Roboflow: {name} -> {out_dir}")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(cfg["workspace"]).project(cfg["project"])
    version = project.version(int(cfg["version"]))
    dataset = version.download(
        cfg.get("format", "yolov8"),
        location=str(out_dir),
    )

    marker.write_text(str(dataset.location), encoding="utf-8")
    print(f"✓ {name}: descargado en {dataset.location}")


def download_with_gdown(file_id: str, output_path: Path) -> bool:
    """
    Descarga desde Google Drive usando gdown.

    Retorna True si la descarga fue exitosa.
    """
    try:
        import gdown
    except Exception:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = gdown.download(
            id=file_id,
            output=str(output_path),
            quiet=False,
            fuzzy=True,
        )
        return result is not None and Path(result).exists()
    except Exception as exc:
        print(f"⚠ gdown falló: {exc}")
        return False


def download_with_url(url: str, output_path: Path) -> bool:
    """
    Descarga básica usando urllib como respaldo.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        urlretrieve(url, output_path)
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception as exc:
        print(f"⚠ Descarga directa falló: {exc}")
        return False


def extract_archive(archive_path: Path, out_dir: Path) -> None:
    """
    Extrae ZIP/TAR si corresponde.

    Si el archivo no es un archivo comprimido reconocido, deja el archivo
    en out_dir para diagnóstico.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(archive_path):
        print(f"Extrayendo ZIP: {archive_path} -> {out_dir}")
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(out_dir)
        return

    if tarfile.is_tarfile(archive_path):
        print(f"Extrayendo TAR: {archive_path} -> {out_dir}")
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(out_dir)
        return

    # Algunos enlaces de Google Drive devuelven un archivo sin extensión.
    # Si no se pudo identificar, se copia para inspección manual.
    fallback = out_dir / archive_path.name
    shutil.copy2(archive_path, fallback)

    raise RuntimeError(
        f"El archivo descargado no parece ZIP/TAR: {archive_path}. "
        f"Se copió a {fallback}. Revisa si Google Drive devolvió una página HTML "
        f"o si el archivo requiere descarga manual."
    )


def download_shwd_from_google_drive(cfg: dict, force: bool = False) -> None:
    """
    Descarga SHWD real desde Google Drive y verifica que contenga XML Pascal VOC.
    """
    out_dir = Path(cfg["out_dir"])
    archive_path = Path(cfg.get("archive_path", "data/raw/shwd_dataset_download"))
    marker = out_dir / ".download_complete"

    if marker.exists() and has_pascal_voc_xml(out_dir) and not force:
        print(f"✓ SHWD: ya descargado y validado en {out_dir}")
        return

    if out_dir.exists() and force:
        shutil.rmtree(out_dir)

    if marker.exists() and not has_pascal_voc_xml(out_dir):
        print("⚠ SHWD tenía marcador de descarga, pero no se encontraron XML.")
        print("  Se eliminará la carpeta incompleta y se descargará nuevamente.")
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    file_id = cfg.get("gdrive_file_id")
    url = cfg.get("gdrive_url")

    if not file_id and not url:
        raise RuntimeError("SHWD requiere gdrive_file_id o gdrive_url en data_sources.yaml.")

    if archive_path.exists() and force:
        archive_path.unlink()

    if not archive_path.exists():
        print(f"Descargando SHWD desde Google Drive -> {archive_path}")

        ok = False

        if file_id:
            ok = download_with_gdown(file_id=file_id, output_path=archive_path)

        if not ok and url:
            ok = download_with_url(url=url, output_path=archive_path)

        if not ok:
            raise RuntimeError(
                "No se pudo descargar SHWD desde Google Drive. "
                "Descárgalo manualmente y colócalo en data/raw/shwd con "
                "Annotations/, ImageSets/ y JPEGImages/."
            )
    else:
        print(f"✓ Archivo SHWD ya existe: {archive_path}")

    extract_archive(archive_path, out_dir)

    if not has_pascal_voc_xml(out_dir):
        diagnostic = "\n".join(str(p) for p in sorted(out_dir.rglob("*"))[:80])
        raise RuntimeError(
            f"SHWD se descargó/extrajo, pero no se encontraron XML Pascal VOC en {out_dir}.\n"
            f"Primeros archivos encontrados:\n{diagnostic}"
        )

    marker.write_text("ok", encoding="utf-8")

    xml_count = len(list(out_dir.rglob("*.xml")))
    print(f"✓ SHWD descargado y validado en {out_dir}")
    print(f"  XML Pascal VOC encontrados: {xml_count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default="configs/data_sources.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with open(args.sources, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    api_key = os.environ.get("ROBOFLOW_API_KEY")

    rf_sources = cfg.get("roboflow", {})
    if rf_sources:
        if not api_key:
            raise RuntimeError(
                "Falta ROBOFLOW_API_KEY. En Colab configura la variable con getpass; "
                "en local usa: export ROBOFLOW_API_KEY='...'."
            )

        for name, src in rf_sources.items():
            if src.get("enabled", True):
                download_roboflow_source(
                    api_key=api_key,
                    name=name,
                    cfg=src,
                    force=args.force,
                )

    shwd = cfg.get("shwd", {})
    if shwd.get("enabled", True):
        download_shwd_from_google_drive(shwd, force=args.force)

    print("\nDescarga de fuentes finalizada.")


if __name__ == "__main__":
    main()