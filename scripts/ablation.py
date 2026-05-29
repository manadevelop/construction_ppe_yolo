#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ablation_yolov8n_no_mosaic.yaml")
    args = parser.parse_args()

    subprocess.run(["python", "scripts/train.py", "--config", args.config], check=True)


if __name__ == "__main__":
    main()
