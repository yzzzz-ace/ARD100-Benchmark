#!/usr/bin/env python3
"""Non-destructive supplemental Ultralytics seed/leakage training."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch
import ultralytics
from ultralytics import RTDETR, YOLO

from public_paths import INITIAL_WEIGHTS_ROOT, REPO_ROOT


ROOT = REPO_ROOT
DEFAULT_DATA = ROOT / "configs" / "ard100_rgb_trainval.yaml"
PROJECT = ROOT / "outputs" / "supplementary" / "training"
YOLOV8_INIT = INITIAL_WEIGHTS_ROOT / "yolov8s.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("yolov8s", "rtdetr_l"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--data-yaml", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not args.data_yaml.exists():
        raise FileNotFoundError(args.data_yaml)
    initialization = YOLOV8_INIT if args.model == "yolov8s" else INITIAL_WEIGHTS_ROOT / "rtdetr-l.pt"
    if not initialization.exists():
        raise FileNotFoundError(initialization)
    model = YOLO(str(initialization)) if args.model == "yolov8s" else RTDETR(str(initialization))
    batch = 64 if args.model == "yolov8s" else 16
    run_name = f"{args.model}_{args.tag}_seed{args.seed}"
    run_dir = PROJECT / run_name
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {run_dir}")
    train_args = dict(
        data=str(args.data_yaml), epochs=args.epochs, patience=10, imgsz=640, batch=batch,
        device=0, workers=args.workers, project=str(PROJECT), name=run_name,
        exist_ok=False, seed=args.seed, deterministic=True, amp=True, cache=False,
        rect=False, plots=True, save=True, save_period=5, pretrained=True,
        cos_lr=False, close_mosaic=0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.5, mosaic=1.0 if args.model == "yolov8s" else 0.0,
        mixup=0.0, copy_paste=0.0, erasing=0.0, val=True, verbose=True,
    )
    if args.model == "yolov8s":
        train_args.update(optimizer="SGD", lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005)
    else:
        train_args.update(
            optimizer="AdamW", lr0=0.0001, lrf=0.01, momentum=0.9,
            weight_decay=0.0001, warmup_bias_lr=0.0,
        )
    model.train(**train_args)
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    if not best.exists() or not last.exists():
        raise RuntimeError(f"Incomplete training output in {run_dir}")
    metadata = {
        "protocol": "ard100-det-supp-v2",
        "status": "PASS",
        "model": args.model,
        "tag": args.tag,
        "seed": args.seed,
        "data_yaml": str(args.data_yaml.resolve()),
        "data_yaml_sha256": sha256(args.data_yaml),
        "initialization": str(initialization.resolve()),
        "initialization_sha256": sha256(initialization),
        "epochs_requested": args.epochs,
        "batch": batch,
        "best": str(best.resolve()),
        "best_sha256": sha256(best),
        "last_sha256": sha256(last),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "train_args": train_args,
    }
    (run_dir / "supplement_manifest.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    print(json.dumps(metadata, indent=2, default=str))


if __name__ == "__main__":
    main()
