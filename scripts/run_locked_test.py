#!/usr/bin/env python3
"""Execute the one-time FP32 official-test suite authorized by test_lock.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from test_lock import LOCK_PATH, PRIMARY_MODELS, load_lock


ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS = {"yolov8s", "rtdetr_l"}
TORCHVISION = {"fasterrcnn_r50_fpn_v2", "retinanet_resnet50_fpn_v2", "fcos_resnet50_fpn"}


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
    script = (
        "infer_ultralytics.py"
        if model_id in ULTRALYTICS
        else "infer_torchvision.py"
        if model_id in TORCHVISION
        else "infer_yolomg.py"
    )
    command = [
        sys.executable,
        str(ROOT / "scripts" / script),
        "--model-id",
        model_id,
        "--weights",
        str(weights),
        "--split",
        "test",
        "--imgsz",
        str(imgsz),
        "--output-id",
        output_id,
    ]
    if model_id in ULTRALYTICS:
        batch = 32 if model_id == "yolov8s" and imgsz == 640 else 16 if imgsz == 640 else 4
        command.extend(["--batch", str(batch)])
    elif model_id in TORCHVISION:
        command.extend(["--batch", "8" if imgsz == 640 else "2"])
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required acknowledgement that this consumes the locked one-time official-test evaluation.",
    )
    args = parser.parse_args()
    if not args.execute:
        parser.error("--execute is required")

    lock = load_lock()
    final_manifest_path = ROOT / "outputs" / "locked_test_manifest.json"
    if final_manifest_path.exists():
        raise FileExistsError(
            f"{final_manifest_path} already exists; the locked primary test must not be rerun silently"
        )

    records: list[dict[str, object]] = []
    for model_id in PRIMARY_MODELS:
        for imgsz in (640, 1280):
            output_id = identifier(model_id, imgsz)
            entry = lock["entries"][output_id]
            weights = Path(entry["weights"])
            prediction_dir = ROOT / "outputs" / "predictions" / output_id
            metric_dir = ROOT / "outputs" / "metrics" / output_id
            prediction_csv = prediction_dir / "test.csv"
            prediction_metadata_path = prediction_dir / "test_metadata.json"
            metrics_path = metric_dir / "test_metrics.json"
            frame_metrics_path = metric_dir / "test_frame_metrics.csv"

            test_artifacts = [prediction_csv, prediction_metadata_path, metrics_path, frame_metrics_path]
            present = [path.exists() for path in test_artifacts]
            if any(present) and not all(present):
                raise RuntimeError(
                    f"Incomplete prior test artifacts for {output_id}; preserve and audit them before continuing"
                )
            if not all(present):
                run(inference_command(model_id, weights, imgsz, output_id))
                run(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "evaluate_predictions.py"),
                        "--model-id",
                        output_id,
                        "--predictions",
                        str(prediction_csv),
                        "--split",
                        "test",
                        "--threshold-file",
                        str(entry["threshold_file"]),
                    ]
                )

            prediction_metadata = json.loads(prediction_metadata_path.read_text(encoding="utf-8"))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if bool(prediction_metadata.get("half", False)):
                raise ValueError(f"{output_id} official-test inference used FP16")
            if int(prediction_metadata["images"]) != 72631 or int(metrics["top1"]["frames"]) != 72631:
                raise ValueError(f"{output_id} official-test denominator mismatch")
            if prediction_metadata["weights_sha256"] != entry["weights_sha256"]:
                raise ValueError(f"{output_id} official-test checkpoint mismatch")
            if float(metrics["threshold"]) != float(entry["threshold"]):
                raise ValueError(f"{output_id} official-test threshold mismatch")
            records.append(
                {
                    "output_id": output_id,
                    "model": model_id,
                    "imgsz": imgsz,
                    "precision": "fp32",
                    "weights_sha256": entry["weights_sha256"],
                    "threshold": entry["threshold"],
                    "prediction_sha256": prediction_metadata["prediction_sha256"],
                    "metrics_sha256": sha256(metrics_path),
                    "frame_metrics_sha256": sha256(frame_metrics_path),
                    "frames": int(metrics["top1"]["frames"]),
                    "AP50_95": float(metrics["average_precision"]["AP50_95"]),
                    "AP50": float(metrics["average_precision"]["AP50"]),
                    "top1_recall_iou50": float(metrics["top1"]["frame_recall_iou50"]),
                }
            )

    payload = {
        "protocol_version": "ard100-det-v1",
        "test_lock": str(LOCK_PATH.resolve()),
        "test_lock_sha256": sha256(LOCK_PATH),
        "one_time_primary_suite_complete": True,
        "entries": records,
    }
    final_manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
