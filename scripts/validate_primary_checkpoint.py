#!/usr/bin/env python3
"""Run the frozen FP32 validation protocol at 640 and 1280 for one checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS = {"yolov8s", "rtdetr_l"}
TORCHVISION = {"fasterrcnn_r50_fpn_v2", "retinanet_resnet50_fpn_v2", "fcos_resnet50_fpn"}
YOLOMG = {"yolomg_paper"}
PRIMARY = ULTRALYTICS | TORCHVISION | YOLOMG


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    print("[RUN] " + subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def identifier(model_id: str, imgsz: int) -> str:
    return model_id if imgsz == 640 else f"{model_id}_i{imgsz}"


def inference_command(model_id: str, weights: Path, imgsz: int, output_id: str) -> list[str]:
    common = [
        sys.executable,
        str(ROOT / "scripts" / (
            "infer_ultralytics.py"
            if model_id in ULTRALYTICS
            else "infer_torchvision.py"
            if model_id in TORCHVISION
            else "infer_yolomg.py"
        )),
        "--model-id",
        model_id,
        "--weights",
        str(weights),
        "--split",
        "val",
        "--imgsz",
        str(imgsz),
        "--output-id",
        output_id,
    ]
    if model_id in ULTRALYTICS:
        batch = 32 if model_id == "yolov8s" and imgsz == 640 else 16 if imgsz == 640 else 4
        common.extend(["--batch", str(batch)])
    elif model_id in TORCHVISION:
        common.extend(["--batch", "8" if imgsz == 640 else "2"])
    return common


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", choices=sorted(PRIMARY), required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--resolutions", type=int, nargs="+", choices=(640, 1280), default=[640, 1280])
    args = parser.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(args.weights)
    weights_hash = sha256(args.weights)
    records: list[dict[str, object]] = []
    for imgsz in args.resolutions:
        output_id = identifier(args.model_id, imgsz)
        prediction_dir = ROOT / "outputs" / "predictions" / output_id
        metric_dir = ROOT / "outputs" / "metrics" / output_id
        if prediction_dir.exists() or metric_dir.exists():
            raise FileExistsError(
                f"Active validation output already exists for {output_id}; "
                "archive an excluded run explicitly instead of overwriting it"
            )
        run(inference_command(args.model_id, args.weights, imgsz, output_id))
        prediction_csv = prediction_dir / "val.csv"
        run(
            [
                sys.executable,
                str(ROOT / "scripts" / "evaluate_predictions.py"),
                "--model-id",
                output_id,
                "--predictions",
                str(prediction_csv),
                "--split",
                "val",
            ]
        )
        prediction_metadata_path = prediction_dir / "val_metadata.json"
        metrics_path = metric_dir / "val_metrics.json"
        threshold_path = metric_dir / "threshold.json"
        prediction_metadata = json.loads(prediction_metadata_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        threshold = json.loads(threshold_path.read_text(encoding="utf-8"))
        if bool(prediction_metadata.get("half", False)):
            raise ValueError(f"{output_id} unexpectedly used FP16")
        if prediction_metadata["weights_sha256"] != weights_hash:
            raise ValueError(f"{output_id} checkpoint hash mismatch")
        if int(prediction_metadata["images"]) != 8938 or int(metrics["top1"]["frames"]) != 8938:
            raise ValueError(f"{output_id} validation denominator mismatch")
        records.append(
            {
                "output_id": output_id,
                "imgsz": imgsz,
                "precision": "fp32",
                "weights_sha256": weights_hash,
                "prediction_sha256": prediction_metadata["prediction_sha256"],
                "threshold": float(threshold["threshold"]),
                "threshold_sha256": sha256(threshold_path),
                "AP50_95": float(metrics["average_precision"]["AP50_95"]),
                "AP50": float(metrics["average_precision"]["AP50"]),
                "top1_recall_iou50": float(metrics["top1"]["frame_recall_iou50"]),
                "macro_video_recall_iou50": float(metrics["top1"]["macro_video_recall_iou50"]),
            }
        )
    output_dir = ROOT / "outputs" / "validation_manifests"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{args.model_id}.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    payload = {
        "protocol_version": "ard100-det-v1",
        "model": args.model_id,
        "weights": str(args.weights.resolve()),
        "weights_sha256": weights_hash,
        "records": records,
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
