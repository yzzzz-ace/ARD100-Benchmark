#!/usr/bin/env python3
"""Run a trained torchvision detector and save post-NMS detections."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from train_torchvision_baseline import build_model, target_label_for_model
from test_lock import require_test_lock

from public_paths import DATA_ROOT, REPO_ROOT


ROOT = REPO_ROOT
DATASET = ROOT / "datasets" / "ARD100_FullFrame_v1"
SOURCE = DATA_ROOT


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def chunks(items: list[Path], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        choices=("fasterrcnn_r50_fpn_v2", "retinanet_resnet50_fpn_v2", "fcos_resnet50_fpn"),
        required=True,
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--imgsz", type=int, choices=(640, 1280), default=640)
    parser.add_argument("--output-id")
    args = parser.parse_args()

    if args.split == "test":
        if args.half:
            raise ValueError("Headline official-test accuracy is frozen to FP32; omit --half")
        require_test_lock(args.model_id, args.weights, args.imgsz)

    device = torch.device("cuda:0")
    model = build_model(args.model_id, pretrained=False, input_size=args.imgsz)
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.to(device).eval()
    target_label = target_label_for_model(args.model_id)
    if args.half:
        model.half()

    image_root = (SOURCE / "images" / "val") if args.split == "val" else (DATASET / "images" / "test")
    image_paths = sorted(image_root.glob("*.jpg"))
    output_id = args.output_id or (args.model_id if args.imgsz == 640 else f"{args.model_id}_i{args.imgsz}")
    output_dir = ROOT / "outputs" / "predictions" / output_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"{args.split}.csv"
    detection_count = image_count = 0
    model_seconds = 0.0
    wall_started = time.perf_counter()

    with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["sample_id", "detection_index", "score", "class_id", "x1", "y1", "x2", "y2"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        with torch.inference_mode():
            for batch_paths in chunks(image_paths, args.batch):
                images = []
                for path in batch_paths:
                    tensor = pil_to_tensor(Image.open(path).convert("RGB")).float().div_(255.0)
                    images.append(tensor.to(device, non_blocking=True).half() if args.half else tensor.to(device, non_blocking=True))
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.amp.autocast("cuda", enabled=args.half, dtype=torch.float16):
                    outputs = model(images)
                torch.cuda.synchronize()
                model_seconds += time.perf_counter() - started
                for path, output in zip(batch_paths, outputs):
                    image_count += 1
                    keep = output["labels"] == target_label
                    boxes = output["boxes"][keep].detach().float().cpu()
                    scores = output["scores"][keep].detach().float().cpu()
                    for detection_index, (box, score) in enumerate(zip(boxes, scores)):
                        writer.writerow(
                            {
                                "sample_id": path.stem,
                                "detection_index": detection_index,
                                "score": float(score),
                                "class_id": target_label,
                                "x1": float(box[0]),
                                "y1": float(box[1]),
                                "x2": float(box[2]),
                                "y2": float(box[3]),
                            }
                        )
                        detection_count += 1
                if image_count % 2000 < args.batch:
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
        "target_label": target_label,
        "model_seconds": model_seconds,
        "model_fps": image_count / model_seconds,
        "wall_seconds": time.perf_counter() - wall_started,
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
