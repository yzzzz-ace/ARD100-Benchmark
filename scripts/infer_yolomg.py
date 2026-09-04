#!/usr/bin/env python3
"""Run YOLOMG-Arch on paired RGB/motion benchmark images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import cv2
import torch

from test_lock import require_test_lock
from public_paths import REPO_ROOT, YOLOMG_CODE_ROOT
from yolomg_adapter import YOLOMGDetector


ROOT = REPO_ROOT
DATASET = ROOT / "datasets" / "ARD100_FullFrame_v1"
DEFAULT_CODE = YOLOMG_CODE_ROOT


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", choices=("yolomg_paper",), required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--code-root", type=Path, default=DEFAULT_CODE)
    parser.add_argument("--imgsz", type=int, choices=(640, 1280), default=640)
    parser.add_argument("--output-id")
    args = parser.parse_args()

    if args.split == "test" and args.model_id == "yolomg_paper":
        require_test_lock(args.model_id, args.weights, args.imgsz)

    detector = YOLOMGDetector(args.code_root, args.weights, args.imgsz, "0", 0.001, 0.60)
    base = ROOT / "datasets" / "ARD100_YOLOMG_Paper_v1"
    rgb_root = base / "images" / args.split
    motion_root = base / "images2" / args.split
    image_paths = sorted(rgb_root.glob("*.jpg"))
    output_id = args.output_id or (args.model_id if args.imgsz == 640 else f"{args.model_id}_i{args.imgsz}")
    output_dir = ROOT / "outputs" / "predictions" / output_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"{args.split}.csv"
    image_count = detection_count = 0
    model_seconds = 0.0
    wall_started = time.perf_counter()
    with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["sample_id", "detection_index", "score", "class_id", "x1", "y1", "x2", "y2"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rgb_path in image_paths:
            motion_path = motion_root / rgb_path.name
            rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            motion = cv2.imread(str(motion_path), cv2.IMREAD_COLOR)
            if rgb is None or motion is None:
                raise OSError(f"Cannot read pair: {rgb_path}, {motion_path}")
            torch.cuda.synchronize()
            started = time.perf_counter()
            detections = detector.detect(rgb, motion)
            torch.cuda.synchronize()
            model_seconds += time.perf_counter() - started
            for detection_index, detection in enumerate(detections):
                writer.writerow(
                    {
                        "sample_id": rgb_path.stem,
                        "detection_index": detection_index,
                        "score": float(detection.conf),
                        "class_id": 0,
                        "x1": float(detection.xyxy[0]),
                        "y1": float(detection.xyxy[1]),
                        "x2": float(detection.xyxy[2]),
                        "y2": float(detection.xyxy[3]),
                    }
                )
                detection_count += 1
            image_count += 1
            if image_count % 2000 == 0:
                print(f"[PROGRESS] {args.model_id} {args.split}: images={image_count}", flush=True)

    metadata = {
        "protocol_version": "ard100-det-v1",
        "model": args.model_id,
        "output_id": output_id,
        "imgsz": args.imgsz,
        "split": args.split,
        "weights": str(args.weights.resolve()),
        "weights_sha256": sha256(args.weights),
        "images": image_count,
        "detections": detection_count,
        "model_seconds": model_seconds,
        "model_fps": image_count / model_seconds,
        "wall_seconds": time.perf_counter() - wall_started,
        "confidence_floor": 0.001,
        "nms_iou": 0.60,
        "gpu": torch.cuda.get_device_name(0),
        "prediction_csv": str(output_csv),
        "prediction_sha256": sha256(output_csv),
    }
    (output_dir / f"{args.split}_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
