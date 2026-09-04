#!/usr/bin/env python3
"""Validation-selected operating points and absent-frame uncertainty.

This is a secondary analysis of immutable prediction CSV files. It does not
rerun inference or replace the locked F1-selected operating point.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate_predictions import ious, load_ground_truth, load_predictions


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "supplementary" / "operating_points"
MODELS = [
    ("fasterrcnn_r50_fpn_v2", "Faster R-CNN"),
    ("retinanet_resnet50_fpn_v2", "RetinaNet"),
    ("fcos_resnet50_fpn", "FCOS"),
    ("yolov8s", "YOLOv8s"),
    ("rtdetr_l", "RT-DETR-L"),
    ("yolomg_paper", "YOLOMG-Arch"),
]
TARGET_FPRS = (0.05, 0.10, 0.20)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def top1_arrays(samples, predictions) -> pd.DataFrame:
    rows = []
    for sample_id, sample in samples.items():
        detections = predictions.get(sample_id, [])
        best = detections[0] if detections else None
        present = len(sample["boxes"]) > 0
        localized = False
        if best is not None and present:
            localized = bool(np.max(ious(best["box"], sample["boxes"])) >= 0.50)
        rows.append(
            {
                "sample_id": sample_id,
                "video_id": sample["video_id"],
                "target_present": int(present),
                "top1_score": float(best["score"]) if best is not None else -np.inf,
                "top1_localizable": int(localized),
            }
        )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame, threshold: float) -> dict[str, float]:
    predicted = frame["top1_score"].to_numpy() >= threshold
    present = frame["target_present"].to_numpy(dtype=bool)
    localized = predicted & frame["top1_localizable"].to_numpy(dtype=bool)
    tp = int(localized.sum())
    fp = int((predicted & ~frame["top1_localizable"].to_numpy(dtype=bool)).sum())
    fn = int((present & ~localized).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(int(present.sum()), 1)
    negatives = ~present
    absent_fpr = float((predicted & negatives).sum() / max(int(negatives.sum()), 1))
    return {
        "threshold": float(threshold),
        "precision_iou50": precision,
        "recall_iou50": recall,
        "f1_iou50": 2 * precision * recall / max(precision + recall, 1e-12),
        "absent_fpr": absent_fpr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "absent_frames": int(negatives.sum()),
    }


def candidates(frame: pd.DataFrame, count: int = 600) -> np.ndarray:
    values = np.unique(frame.loc[np.isfinite(frame["top1_score"]), "top1_score"].to_numpy())
    values.sort()
    if len(values) > count:
        values = values[np.unique(np.linspace(0, len(values) - 1, count).astype(int))]
    return np.unique(np.concatenate(([0.001], values, [0.999999])))


def select_fixed_fpr(frame: pd.DataFrame, target: float) -> dict[str, float]:
    summaries = [summarize(frame, float(value)) for value in candidates(frame, 2000)]
    eligible = [row for row in summaries if row["absent_fpr"] <= target + 1e-12]
    if not eligible:
        return min(summaries, key=lambda row: row["absent_fpr"])
    return max(eligible, key=lambda row: (row["recall_iou50"], row["precision_iou50"], -row["threshold"]))


def cluster_bootstrap_absent_fpr(frame: pd.DataFrame, threshold: float, resamples: int = 10000) -> tuple[float, float]:
    negative = frame[frame["target_present"] == 0].copy()
    negative["positive"] = (negative["top1_score"] >= threshold).astype(int)
    grouped = negative.groupby("video_id")["positive"].agg(["sum", "count"])
    if len(grouped) == 0:
        return float("nan"), float("nan")
    values = grouped[["sum", "count"]].to_numpy(dtype=float)
    rng = np.random.default_rng(20260828)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    sampled = values[indices]
    draws = sampled[:, :, 0].sum(axis=1) / sampled[:, :, 1].sum(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    split_frames: dict[tuple[str, str], pd.DataFrame] = {}
    input_hashes = {}
    for split in ("val", "test"):
        samples, _ = load_ground_truth(split)
        for model_id, _display in MODELS:
            prediction_path = ROOT / "outputs" / "predictions" / model_id / f"{split}.csv"
            input_hashes[str(prediction_path.relative_to(ROOT))] = sha256(prediction_path)
            split_frames[(model_id, split)] = top1_arrays(samples, load_predictions(prediction_path))

    records = []
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = plt.get_cmap("tab10")
    for model_index, (model_id, display) in enumerate(MODELS):
        val = split_frames[(model_id, "val")]
        test = split_frames[(model_id, "test")]
        primary = json.loads((ROOT / "outputs" / "metrics" / model_id / "threshold.json").read_text())
        primary_threshold = float(primary["threshold"])
        primary_test = summarize(test, primary_threshold)
        low, high = cluster_bootstrap_absent_fpr(test, primary_threshold)
        records.append(
            {
                "model": model_id,
                "display": display,
                "operating_point": "validation_F1",
                "target_validation_absent_fpr": np.nan,
                **{f"validation_{k}": v for k, v in summarize(val, primary_threshold).items()},
                **{f"test_{k}": v for k, v in primary_test.items()},
                "test_absent_fpr_cluster_ci95_low": low,
                "test_absent_fpr_cluster_ci95_high": high,
            }
        )
        for target in TARGET_FPRS:
            chosen = select_fixed_fpr(val, target)
            test_summary = summarize(test, chosen["threshold"])
            low, high = cluster_bootstrap_absent_fpr(test, chosen["threshold"])
            records.append(
                {
                    "model": model_id,
                    "display": display,
                    "operating_point": f"validation_absent_FPR<={target:.2f}",
                    "target_validation_absent_fpr": target,
                    **{f"validation_{k}": v for k, v in chosen.items()},
                    **{f"test_{k}": v for k, v in test_summary.items()},
                    "test_absent_fpr_cluster_ci95_low": low,
                    "test_absent_fpr_cluster_ci95_high": high,
                }
            )

        curve = pd.DataFrame([summarize(test, float(value)) for value in candidates(test)])
        curve.to_csv(OUT / f"{model_id}_test_froc.csv", index=False)
        ax.plot(curve["absent_fpr"], curve["recall_iou50"], label=display, color=colors(model_index))
        ax.scatter([primary_test["absent_fpr"]], [primary_test["recall_iou50"]], color=colors(model_index), s=22)

    table = pd.DataFrame(records)
    table.to_csv(OUT / "operating_points.csv", index=False, encoding="utf-8-sig")
    ax.set_xlabel("Target-absent frame false-positive rate")
    ax.set_ylabel("Target-present top-1 IoU50 recall")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "test_froc.png", dpi=240)
    fig.savefig(OUT / "test_froc.pdf")
    plt.close(fig)

    manifest = {
        "protocol": "ard100-det-supp-v2",
        "status": "PASS",
        "selection": "validation-only fixed absent-FPR thresholds",
        "test_sweep_role": "descriptive FROC only",
        "target_absent_fprs": TARGET_FPRS,
        "bootstrap": "10,000 resamples of the 15 test videos containing absent frames",
        "input_hashes": input_hashes,
        "outputs": ["operating_points.csv", "test_froc.png", "test_froc.pdf"],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
