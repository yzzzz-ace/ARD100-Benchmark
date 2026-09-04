#!/usr/bin/env python3
"""Run an Ultralytics detector and save every post-NMS detection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import torch
from ultralytics import RTDETR, YOLO

from test_lock import require_test_lock

from public_paths import DATA_ROOT, REPO_ROOT


ROOT = REPO_ROOT
DATASET = ROOT / "datasets" / "ARD100_FullFrame_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", choices=("yolov8s", "rtdetr_l"), required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--imgsz", type=int, choices=(640, 1280), default=640)
    parser.add_argument("--output-id")
    args = parser.parse_args()

    if args.split == "test":
        if args.half:
            raise ValueError("Headline official-test accuracy is frozen to FP32; omit --half")
        require_test_lock(args.model_id, args.weights, args.imgsz)

    image_root = (
        DATA_ROOT / "images" / args.split
        if args.split == "val"
        else DATASET / "images" / "test"
    )
    output_id = args.output_id or (args.model_id if args.imgsz == 640 else f"{args.model_id}_i{args.imgsz}")
    output_dir = ROOT / "outputs" / "predictions" / output_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"{args.split}.csv"
    model = RTDETR(str(args.weights)) if args.model_id == "rtdetr_l" else YOLO(str(args.weights))

    started = time.perf_counter()
    image_count = detection_count = 0
    inference_ms = preprocess_ms = postprocess_ms = 0.0
    with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["sample_id", "detection_index", "score", "class_id", "x1", "y1", "x2", "y2"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        stream = model.predict(
            source=str(image_root),
            stream=True,
            batch=args.batch,
            imgsz=args.imgsz,
            conf=0.001,
            iou=0.60,
            max_det=300,
            device=0,
            half=args.half,
            verbose=False,
            save=False,
        )
        for result in stream:
            sample_id = Path(result.path).stem
            image_count += 1
            preprocess_ms += float(result.speed.get("preprocess", 0.0))
            inference_ms += float(result.speed.get("inference", 0.0))
            postprocess_ms += float(result.speed.get("postprocess", 0.0))
            boxes = result.boxes
            if boxes is None:
                continue
            for detection_index, (xyxy, score, class_id) in enumerate(
                zip(boxes.xyxy.detach().cpu(), boxes.conf.detach().cpu(), boxes.cls.detach().cpu())
            ):
                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "detection_index": detection_index,
                        "score": float(score),
                        "class_id": int(class_id),
                        "x1": float(xyxy[0]),
                        "y1": float(xyxy[1]),
                        "x2": float(xyxy[2]),
                        "y2": float(xyxy[3]),
                    }
                )
                detection_count += 1
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
        "half": args.half,
        "batch": args.batch,
        "wall_seconds": time.perf_counter() - started,
        "mean_preprocess_ms": preprocess_ms / max(image_count, 1),
        "mean_inference_ms": inference_ms / max(image_count, 1),
        "mean_postprocess_ms": postprocess_ms / max(image_count, 1),
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
