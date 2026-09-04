#!/usr/bin/env python3
"""Unified AP, frame-level, video-macro, and factorized evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from test_lock import require_test_threshold
from PIL import Image

from public_paths import DATA_ROOT, REPO_ROOT


ROOT = REPO_ROOT
DATASET = ROOT / "datasets" / "ARD100_FullFrame_v1"
SOURCE = DATA_ROOT


def load_ground_truth(split: str) -> tuple[dict[str, dict[str, object]], pd.DataFrame | None]:
    if split == "test":
        metadata = pd.read_csv(DATASET / "metadata" / "test_frames.csv")
        samples: dict[str, dict[str, object]] = {}
        for row in metadata.itertuples(index=False):
            scale = float(row.stored_scale)
            boxes = []
            if int(row.target_present):
                boxes.append(
                    [
                        float(row.target_xmin) * scale,
                        float(row.target_ymin) * scale,
                        float(row.target_xmax) * scale,
                        float(row.target_ymax) * scale,
                    ]
                )
            samples[str(row.sample_id)] = {
                "video_id": str(row.video_id),
                "width": int(row.stored_width),
                "height": int(row.stored_height),
                "scale": scale,
                "boxes": np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
            }
        return samples, metadata
    else:
        image_root = SOURCE / "images" / "val"
        label_root = SOURCE / "labels" / "val"
        metadata = None

    samples: dict[str, dict[str, object]] = {}
    for image_path in sorted(image_root.glob("*.jpg")):
        sample_id = image_path.stem
        with Image.open(image_path) as image:
            width, height = image.size
        scale = 1.0
        video_id = sample_id.rsplit("_", 1)[0]
        boxes: list[list[float]] = []
        label_path = label_root / f"{sample_id}.txt"
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 5:
                continue
            _, cx, cy, box_width, box_height = map(float, fields)
            bw, bh = box_width * width, box_height * height
            boxes.append([cx * width - bw / 2, cy * height - bh / 2, cx * width + bw / 2, cy * height + bh / 2])
        samples[sample_id] = {
            "video_id": video_id,
            "width": width,
            "height": height,
            "scale": scale,
            "boxes": np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
        }
    return samples, metadata


def load_predictions(path: Path) -> dict[str, list[dict[str, float]]]:
    predictions: dict[str, list[dict[str, float]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            predictions[row["sample_id"]].append(
                {
                    "score": float(row["score"]),
                    "box": np.asarray([float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])], dtype=np.float64),
                }
            )
    for detections in predictions.values():
        detections.sort(key=lambda item: item["score"], reverse=True)
    return predictions


def ious(box: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if len(targets) == 0:
        return np.empty((0,), dtype=np.float64)
    top_left = np.maximum(box[:2], targets[:, :2])
    bottom_right = np.minimum(box[2:], targets[:, 2:])
    intersection = np.maximum(bottom_right - top_left, 0.0).prod(axis=1)
    box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    target_area = np.maximum(targets[:, 2] - targets[:, 0], 0.0) * np.maximum(targets[:, 3] - targets[:, 1], 0.0)
    return intersection / np.maximum(box_area + target_area - intersection, 1e-12)


def average_precision(
    samples: dict[str, dict[str, object]],
    predictions: dict[str, list[dict[str, float]]],
    iou_threshold: float,
    include_samples: set[str] | None = None,
) -> float:
    scored: list[tuple[float, int]] = []
    positives = 0
    selected = include_samples if include_samples is not None else set(samples)
    for sample_id in selected:
        gt = samples[sample_id]["boxes"]
        positives += len(gt)
        used: set[int] = set()
        for detection in predictions.get(sample_id, []):
            overlap = ious(detection["box"], gt)
            if len(overlap):
                best_index = int(np.argmax(overlap))
                matched = overlap[best_index] >= iou_threshold and best_index not in used
            else:
                best_index, matched = -1, False
            scored.append((detection["score"], int(matched)))
            if matched:
                used.add(best_index)
    if positives == 0 or not scored:
        return 0.0
    scored.sort(key=lambda item: item[0], reverse=True)
    truth = np.asarray([item[1] for item in scored], dtype=np.float64)
    tp = np.cumsum(truth)
    fp = np.cumsum(1.0 - truth)
    recall = tp / positives
    precision = tp / np.maximum(tp + fp, 1e-12)
    values = [precision[recall >= level].max() if np.any(recall >= level) else 0.0 for level in np.linspace(0.0, 1.0, 101)]
    return float(np.mean(values))


def top1_rows(
    samples: dict[str, dict[str, object]],
    predictions: dict[str, list[dict[str, float]]],
    threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample_id, sample in samples.items():
        gt = sample["boxes"]
        detections = [item for item in predictions.get(sample_id, []) if item["score"] >= threshold]
        best = detections[0] if detections else None
        present = len(gt) > 0
        if best is not None and present:
            overlap_values = ious(best["box"], gt)
            best_gt_index = int(np.argmax(overlap_values))
            overlap = float(overlap_values[best_gt_index])
            predicted_center = (best["box"][:2] + best["box"][2:]) / 2.0
            gt_center = (gt[best_gt_index, :2] + gt[best_gt_index, 2:]) / 2.0
            center_error = float(np.linalg.norm(predicted_center - gt_center) / float(sample["scale"]))
        else:
            overlap = 0.0
            center_error = float("inf")
        localized = present and best is not None and overlap >= 0.50
        frame_fp = (best is not None and not present) or (best is not None and present and not localized)
        frame_fn = present and not localized
        absent_false_positive_detections = len(detections) if not present else 0
        rows.append(
            {
                "sample_id": sample_id,
                "video_id": sample["video_id"],
                "target_present": int(present),
                "prediction_present": int(best is not None),
                "localized_iou50": int(localized),
                "frame_fp": int(frame_fp),
                "frame_fn": int(frame_fn),
                "detections_at_threshold": len(detections),
                "absent_false_positive_detections": absent_false_positive_detections,
                "top1_iou": overlap,
                "center_error_original_px": center_error,
                "top1_score": best["score"] if best is not None else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_top1(frame: pd.DataFrame) -> dict[str, float]:
    positives = int(frame["target_present"].sum())
    negatives = len(frame) - positives
    tp = int(frame["localized_iou50"].sum())
    fp = int(frame["frame_fp"].sum())
    fn = int(frame["frame_fn"].sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    present = frame[frame["target_present"] == 1]
    finite = present[np.isfinite(present["center_error_original_px"])]
    per_video = frame.groupby("video_id", sort=True).apply(
        lambda part: pd.Series(
            {
                "positive_frames": int(part["target_present"].sum()),
                "localized_frames": int(part["localized_iou50"].sum()),
                "recall_iou50": part["localized_iou50"].sum() / max(part["target_present"].sum(), 1),
                "absent_frames": int((part["target_present"] == 0).sum()),
                "false_positive_absent_frames": int(((part["target_present"] == 0) & (part["prediction_present"] == 1)).sum()),
                "false_positive_absent_detections": int(part["absent_false_positive_detections"].sum()),
            }
        ),
        include_groups=False,
    ).reset_index()
    return {
        "frames": len(frame),
        "positive_frames": positives,
        "negative_frames": negatives,
        "true_localizations_iou50": tp,
        "frame_false_positives": fp,
        "frame_false_negatives": fn,
        "frame_precision_iou50": precision,
        "frame_recall_iou50": recall,
        "frame_f1_iou50": f1,
        "macro_video_recall_iou50": float(per_video["recall_iou50"].mean()),
        "top1_mean_iou_present": float(present["top1_iou"].mean()),
        "center_precision_5px": float((finite["center_error_original_px"] <= 5).sum() / max(len(present), 1)),
        "center_precision_10px": float((finite["center_error_original_px"] <= 10).sum() / max(len(present), 1)),
        "center_precision_20px": float((finite["center_error_original_px"] <= 20).sum() / max(len(present), 1)),
        "false_positive_frame_rate_absent": float(
            ((frame["target_present"] == 0) & (frame["prediction_present"] == 1)).sum() / max(negatives, 1)
        ),
        "false_positive_detections_per_absent_frame": float(
            frame["absent_false_positive_detections"].sum() / max(negatives, 1)
        ),
    }


def size_conditioned_average_precision(
    samples: dict[str, dict[str, object]],
    predictions: dict[str, list[dict[str, float]]],
    metadata: pd.DataFrame,
) -> dict[str, dict[str, float | int]]:
    positive = metadata[metadata["target_present"] == 1].copy()
    positive["size_bin"] = pd.cut(
        positive["target_sqrt_area_px"],
        bins=[-np.inf, 12, 20, 32, np.inf],
        labels=["<=12", "12-20", "20-32", ">32"],
    )
    output: dict[str, dict[str, float | int]] = {}
    for size_bin, part in positive.groupby("size_bin", observed=True):
        include = set(part["sample_id"].astype(str))
        values = {
            f"AP{int(threshold * 100)}": average_precision(samples, predictions, threshold, include)
            for threshold in np.arange(0.50, 0.951, 0.05)
        }
        output[str(size_bin)] = {
            "frames": len(include),
            "AP50": values["AP50"],
            "AP75": values["AP75"],
            "AP50_95": float(np.mean(list(values.values()))),
        }
    return output


def calibrate_threshold(
    samples: dict[str, dict[str, object]], predictions: dict[str, list[dict[str, float]]]
) -> tuple[float, dict[str, float]]:
    sample_ids = list(samples)
    top_scores_array = np.asarray(
        [predictions[sample_id][0]["score"] if predictions.get(sample_id) else -np.inf for sample_id in sample_ids],
        dtype=np.float64,
    )
    target_present = np.asarray([len(samples[sample_id]["boxes"]) > 0 for sample_id in sample_ids], dtype=bool)
    localized_at_any_score = np.zeros(len(sample_ids), dtype=bool)
    for index, sample_id in enumerate(sample_ids):
        detections = predictions.get(sample_id, [])
        if detections and target_present[index]:
            localized_at_any_score[index] = bool(
                np.max(ious(detections[0]["box"], samples[sample_id]["boxes"])) >= 0.50
            )

    top_scores = sorted(set(top_scores_array[np.isfinite(top_scores_array)].tolist()), reverse=True)
    if len(top_scores) > 1000:
        indices = np.linspace(0, len(top_scores) - 1, 1000).astype(int)
        candidates = [top_scores[index] for index in indices]
    else:
        candidates = top_scores
    candidates = sorted(set([0.001, 0.999] + candidates))
    best_threshold, best_summary = 0.001, {}
    best_key = (-1.0, -1.0, -1.0)
    for threshold in candidates:
        prediction_present = top_scores_array >= threshold
        localized = prediction_present & localized_at_any_score
        tp = int(localized.sum())
        fp = int((prediction_present & ~localized_at_any_score).sum())
        fn = int((target_present & ~localized).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        key = (f1, precision, recall)
        if key > best_key:
            best_key = key
            best_threshold = threshold
    best_summary = summarize_top1(top1_rows(samples, predictions, best_threshold))
    return float(best_threshold), best_summary


def factor_analysis(frame: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    joined = frame.merge(metadata, on=["sample_id", "video_id", "target_present"], how="left")
    positive = joined[joined["target_present"] == 1].copy()
    positive["size_bin"] = pd.cut(
        positive["target_sqrt_area_px"],
        bins=[-np.inf, 12, 20, 32, np.inf],
        labels=["<=12", "12-20", "20-32", ">32"],
    )
    factor_columns = [
        "target_sqrt_area_px",
        "target_contrast",
        "target_context_laplacian_var",
        "context_edge_density",
        "target_displacement_per_frame",
        "camera_displacement_px",
    ]
    records: list[dict[str, object]] = []
    for factor in factor_columns:
        subset = positive[np.isfinite(positive[factor])].copy()
        if factor == "target_sqrt_area_px":
            subset["factor_bin"] = subset["size_bin"].astype(str)
        else:
            subset["factor_bin"] = pd.qcut(subset[factor], q=3, labels=["low", "medium", "high"], duplicates="drop").astype(str)
        for factor_bin, part in subset.groupby("factor_bin", observed=True):
            records.append(
                {
                    "factor": factor,
                    "bin": factor_bin,
                    "frames": len(part),
                    "value_min": float(part[factor].min()),
                    "value_median": float(part[factor].median()),
                    "value_max": float(part[factor].max()),
                    "localization_recall_iou50": float(part["localized_iou50"].mean()),
                    "top1_mean_iou": float(part["top1_iou"].mean()),
                    "center_precision_10px": float((part["center_error_original_px"] <= 10).mean()),
                }
            )
    return pd.DataFrame(records)


def bootstrap_macro_video_recall(frame: pd.DataFrame, resamples: int = 10000) -> dict[str, float]:
    per_video = frame.groupby("video_id").apply(
        lambda part: part["localized_iou50"].sum() / max(part["target_present"].sum(), 1),
        include_groups=False,
    ).to_numpy(dtype=float)
    rng = np.random.default_rng(20260821)
    draws = rng.choice(per_video, size=(resamples, len(per_video)), replace=True).mean(axis=1)
    return {
        "estimate": float(per_video.mean()),
        "bootstrap95_low": float(np.quantile(draws, 0.025)),
        "bootstrap95_high": float(np.quantile(draws, 0.975)),
        "resamples": resamples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--threshold-file", type=Path)
    parser.add_argument("--include-samples", type=Path)
    args = parser.parse_args()

    if args.split == "test":
        if args.threshold_file is None:
            raise ValueError("Test evaluation requires --threshold-file from validation")
        require_test_threshold(args.model_id, args.threshold_file)

    samples, metadata = load_ground_truth(args.split)
    if args.include_samples is not None:
        include_ids = {
            line.strip() for line in args.include_samples.read_text(encoding="utf-8-sig").splitlines() if line.strip()
        }
        missing_ids = include_ids - set(samples)
        if missing_ids:
            raise ValueError(f"Include list contains {len(missing_ids)} unknown sample IDs")
        samples = {sample_id: samples[sample_id] for sample_id in sorted(include_ids)}
        if metadata is not None:
            metadata = metadata[metadata["sample_id"].isin(include_ids)].copy()
    predictions = load_predictions(args.predictions)
    unknown = set(predictions) - set(samples)
    if unknown:
        raise ValueError(f"Predictions contain {len(unknown)} unknown sample IDs")
    output_dir = ROOT / "outputs" / "metrics" / args.model_id
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.split == "val" and args.threshold is None and args.threshold_file is None:
        threshold, calibration = calibrate_threshold(samples, predictions)
        threshold_payload = {
            "protocol_version": "ard100-det-v1",
            "model": args.model_id,
            "source_split": "val",
            "threshold": threshold,
            "selection_metric": "maximum frame F1 at IoU>=0.50",
            "validation_summary": calibration,
        }
        threshold_path = output_dir / "threshold.json"
        threshold_path.write_text(json.dumps(threshold_payload, indent=2) + "\n", encoding="utf-8")
    elif args.threshold_file is not None:
        threshold_payload = json.loads(args.threshold_file.read_text(encoding="utf-8"))
        if threshold_payload.get("source_split") != "val":
            raise ValueError("Threshold file was not calibrated on validation")
        threshold = float(threshold_payload["threshold"])
    elif args.threshold is not None:
        threshold = float(args.threshold)
    else:
        raise ValueError("Test evaluation requires --threshold-file from validation")

    ap = {f"AP{int(value * 100)}": average_precision(samples, predictions, value) for value in np.arange(0.50, 0.951, 0.05)}
    ap["AP50_95"] = float(np.mean(list(ap.values())))
    frame = top1_rows(samples, predictions, threshold)
    summary = summarize_top1(frame)
    result = {
        "protocol_version": "ard100-det-v1",
        "model": args.model_id,
        "split": args.split,
        "threshold": threshold,
        "average_precision": ap,
        "top1": summary,
    }
    if args.split == "test":
        result["macro_video_recall_bootstrap"] = bootstrap_macro_video_recall(frame)
        if metadata is None:
            raise AssertionError("Test metadata missing")
        result["size_conditioned_average_precision"] = size_conditioned_average_precision(
            samples, predictions, metadata
        )
        factors = factor_analysis(frame, metadata)
        factors.to_csv(output_dir / "test_factor_metrics.csv", index=False, encoding="utf-8-sig")
    frame.to_csv(output_dir / f"{args.split}_frame_metrics.csv", index=False, encoding="utf-8-sig")
    (output_dir / f"{args.split}_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
