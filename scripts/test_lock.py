#!/usr/bin/env python3
"""Create and enforce the one-time official-test authorization manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "configs" / "test_lock.json"
PRIMARY_MODELS = (
    "fasterrcnn_r50_fpn_v2",
    "retinanet_resnet50_fpn_v2",
    "fcos_resnet50_fpn",
    "yolov8s",
    "rtdetr_l",
    "yolomg_paper",
)
RESOLUTIONS = (640, 1280)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def output_id(model_id: str, imgsz: int) -> str:
    return model_id if imgsz == 640 else f"{model_id}_i{imgsz}"


def split_output_id(identifier: str) -> tuple[str, int]:
    if identifier.endswith("_i1280"):
        return identifier[:-6], 1280
    return identifier, 640


def load_lock() -> dict[str, object]:
    if not LOCK_PATH.exists():
        raise RuntimeError(
            f"Official test is locked: {LOCK_PATH} does not exist. "
            "Freeze all six validation-selected checkpoints and both resolution thresholds first."
        )
    payload = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if payload.get("protocol_version") != "ard100-det-v1":
        raise ValueError("Unexpected test-lock protocol version")
    return payload


def require_test_lock(model_id: str, weights: Path, imgsz: int) -> dict[str, object]:
    payload = load_lock()
    key = output_id(model_id, imgsz)
    entries = payload.get("entries", {})
    if key not in entries:
        raise RuntimeError(f"Official test is not authorized for {key}")
    entry = entries[key]
    actual_hash = sha256(weights)
    if actual_hash != entry["weights_sha256"]:
        raise ValueError(
            f"Checkpoint hash mismatch for {key}: lock={entry['weights_sha256']} actual={actual_hash}"
        )
    return entry


def require_test_threshold(identifier: str, threshold_file: Path) -> dict[str, object] | None:
    model_id, imgsz = split_output_id(identifier)
    if model_id not in PRIMARY_MODELS:
        return None
    payload = load_lock()
    key = output_id(model_id, imgsz)
    entry = payload["entries"].get(key)
    if entry is None:
        raise RuntimeError(f"Official test threshold is not authorized for {key}")
    actual_hash = sha256(threshold_file)
    if actual_hash != entry["threshold_sha256"]:
        raise ValueError(
            f"Validation-threshold hash mismatch for {key}: "
            f"lock={entry['threshold_sha256']} actual={actual_hash}"
        )
    return entry


def build_lock() -> dict[str, object]:
    if LOCK_PATH.exists():
        raise FileExistsError(
            f"{LOCK_PATH} already exists. A frozen official-test manifest must not be overwritten."
        )
    entries: dict[str, dict[str, object]] = {}
    for model_id in PRIMARY_MODELS:
        weights = ROOT / "weights" / f"{model_id}_ard100_det_v1_best.pt"
        if not weights.exists():
            raise FileNotFoundError(weights)
        weights_hash = sha256(weights)
        for imgsz in RESOLUTIONS:
            identifier = output_id(model_id, imgsz)
            metric_dir = ROOT / "outputs" / "metrics" / identifier
            prediction_dir = ROOT / "outputs" / "predictions" / identifier
            threshold_path = metric_dir / "threshold.json"
            metrics_path = metric_dir / "val_metrics.json"
            prediction_metadata_path = prediction_dir / "val_metadata.json"
            for required in (threshold_path, metrics_path, prediction_metadata_path):
                if not required.exists():
                    raise FileNotFoundError(required)
            threshold = json.loads(threshold_path.read_text(encoding="utf-8"))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            prediction_metadata = json.loads(prediction_metadata_path.read_text(encoding="utf-8"))
            if threshold.get("source_split") != "val" or metrics.get("split") != "val":
                raise ValueError(f"{identifier} is not validation-derived")
            if int(metrics["top1"]["frames"]) != 8938 or int(prediction_metadata["images"]) != 8938:
                raise ValueError(f"{identifier} validation denominator mismatch")
            if int(prediction_metadata["imgsz"]) != imgsz:
                raise ValueError(f"{identifier} input-size mismatch")
            if bool(prediction_metadata.get("half", False)):
                raise ValueError(f"{identifier} accuracy predictions are FP16; FP32 is required")
            if prediction_metadata["weights_sha256"] != weights_hash:
                raise ValueError(f"{identifier} validation predictions use a different checkpoint")
            entries[identifier] = {
                "model": model_id,
                "imgsz": imgsz,
                "weights": str(weights.resolve()),
                "weights_sha256": weights_hash,
                "threshold": float(threshold["threshold"]),
                "threshold_file": str(threshold_path.resolve()),
                "threshold_sha256": sha256(threshold_path),
                "validation_metrics_file": str(metrics_path.resolve()),
                "validation_metrics_sha256": sha256(metrics_path),
                "validation_prediction_metadata_file": str(prediction_metadata_path.resolve()),
                "validation_prediction_metadata_sha256": sha256(prediction_metadata_path),
            }
    return {
        "protocol_version": "ard100-det-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "one-time official-test authorization after validation-only freezing",
        "models": list(PRIMARY_MODELS),
        "resolutions": list(RESOLUTIONS),
        "entries": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="Create configs/test_lock.json after all 12 model-resolution validation results exist.",
    )
    args = parser.parse_args()
    if not args.freeze:
        parser.error("--freeze is required; lock creation is an explicit one-time action")
    payload = build_lock()
    LOCK_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"test_lock": str(LOCK_PATH), "entries": len(payload["entries"])}, indent=2))


if __name__ == "__main__":
    main()
