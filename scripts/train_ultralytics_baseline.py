#!/usr/bin/env python3
"""Train a controlled Ultralytics baseline on the frozen ARD100 split."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
from pathlib import Path

import torch
import ultralytics
from ultralytics import RTDETR, YOLO

from public_paths import INITIAL_WEIGHTS_ROOT, REPO_ROOT


ROOT = REPO_ROOT
DATA_YAML = ROOT / "configs" / "ard100_rgb_trainval.yaml"
PROJECT = ROOT / "outputs" / "training" / "ultralytics"
YOLOV8_INIT = INITIAL_WEIGHTS_ROOT / "yolov8s.pt"
SEED = 20260821


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_spec(model_id: str) -> tuple[object, str, int]:
    if model_id == "yolov8s":
        if not YOLOV8_INIT.exists():
            raise FileNotFoundError(YOLOV8_INIT)
        return YOLO(str(YOLOV8_INIT)), str(YOLOV8_INIT), 64
    if model_id == "rtdetr_l":
        checkpoint = "rtdetr-l.pt"
        return RTDETR(checkpoint), checkpoint, 16
    raise ValueError(model_id)


def abort_on_nonfinite_loss(trainer: object) -> None:
    """Stop immediately instead of allowing a NaN checkpoint to propagate."""
    training_loss = getattr(trainer, "tloss", None)
    if training_loss is not None and not torch.isfinite(training_loss).all():
        raise FloatingPointError(
            f"Non-finite training loss at epoch={getattr(trainer, 'epoch', 'unknown')} "
            f"batch={getattr(trainer, 'batch_i', 'unknown')}: {training_loss}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("yolov8s", "rtdetr_l"), required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    model, initialization, default_batch = model_spec(args.model)
    batch = args.batch or default_batch
    optimizer_tag = "sgd" if args.model == "yolov8s" else "adamw"
    run_name = f"{args.model}_ard100_det_v1_nativeaug_{optimizer_tag}_e{args.epochs}_b{batch}"
    PROJECT.mkdir(parents=True, exist_ok=True)

    train_args = dict(
        data=str(DATA_YAML),
        epochs=args.epochs,
        patience=10,
        imgsz=640,
        batch=batch,
        device=0,
        workers=args.workers,
        project=str(PROJECT),
        name=run_name,
        exist_ok=args.resume,
        seed=SEED,
        deterministic=True,
        amp=True,
        cache=False,
        rect=False,
        plots=True,
        save=True,
        save_period=5,
        pretrained=True,
        cos_lr=False,
        close_mosaic=0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0 if args.model == "yolov8s" else 0.0,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.0,
        val=True,
        verbose=True,
    )
    if args.model == "yolov8s":
        train_args.update(
            optimizer="SGD",
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
        )
    else:
        train_args.update(
            optimizer="AdamW",
            lr0=0.0001,
            lrf=0.01,
            momentum=0.9,
            weight_decay=0.0001,
            warmup_bias_lr=0.0,
        )
    if args.resume:
        train_args["resume"] = True
    model.add_callback("on_train_batch_end", abort_on_nonfinite_loss)
    model.train(**train_args)

    run_dir = PROJECT / run_name
    weights_dir = run_dir / "weights"
    best = weights_dir / "best.pt"
    last = weights_dir / "last.pt"
    if not best.exists() or not last.exists():
        raise RuntimeError(f"Training completed without expected weights in {weights_dir}")

    frozen_dir = ROOT / "weights"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    frozen_best = frozen_dir / f"{args.model}_ard100_det_v1_best.pt"
    frozen_last = frozen_dir / f"{args.model}_ard100_det_v1_last.pt"
    shutil.copy2(best, frozen_best)
    shutil.copy2(last, frozen_last)
    metadata = {
        "protocol_version": "ard100-det-v1",
        "model": args.model,
        "initialization": initialization,
        "epochs_requested": args.epochs,
        "batch": batch,
        "seed": SEED,
        "data_yaml": str(DATA_YAML),
        "data_yaml_sha256": sha256(DATA_YAML),
        "best": str(frozen_best),
        "best_sha256": sha256(frozen_best),
        "last": str(frozen_last),
        "last_sha256": sha256(frozen_last),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "train_args": train_args,
    }
    (frozen_dir / f"{args.model}_ard100_det_v1.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
