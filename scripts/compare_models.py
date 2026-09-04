#!/usr/bin/env python3
"""Assemble the locked test table and paired video-level comparisons."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRICS_ROOT = ROOT / "outputs" / "metrics"
OUTPUT_ROOT = ROOT / "outputs" / "comparisons"
DEFAULT_MODELS = [
    "fasterrcnn_r50_fpn_v2",
    "retinanet_resnet50_fpn_v2",
    "fcos_resnet50_fpn",
    "yolov8s",
    "rtdetr_l",
    "yolomg_paper",
]


def per_video_recall(frame: pd.DataFrame) -> pd.Series:
    grouped = frame.groupby("video_id", sort=True)
    return grouped.apply(
        lambda part: part["localized_iou50"].sum() / max(part["target_present"].sum(), 1),
        include_groups=False,
    )


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p_values[int(index)])
        adjusted[int(index)] = min(running, 1.0)
    return adjusted.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--comparison-id", default="main_i640")
    args = parser.parse_args()

    output_dir = OUTPUT_ROOT / args.comparison_id
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    video_results: dict[str, pd.Series] = {}
    for model_id in args.models:
        metrics_path = METRICS_ROOT / model_id / "test_metrics.json"
        frame_path = METRICS_ROOT / model_id / "test_frame_metrics.csv"
        if not metrics_path.exists() or not frame_path.exists():
            raise FileNotFoundError(f"Missing locked test result for {model_id}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        frame = pd.read_csv(frame_path)
        if int(metrics["top1"]["frames"]) != len(frame):
            raise ValueError(f"Frame-count mismatch for {model_id}")
        video_results[model_id] = per_video_recall(frame)
        bootstrap = metrics["macro_video_recall_bootstrap"]
        summaries.append(
            {
                "model": model_id,
                "threshold_from_val": metrics["threshold"],
                "AP50_95": metrics["average_precision"]["AP50_95"],
                "AP50": metrics["average_precision"]["AP50"],
                "AP75": metrics["average_precision"]["AP75"],
                "top1_frame_recall_iou50": metrics["top1"]["frame_recall_iou50"],
                "macro_video_recall_iou50": metrics["top1"]["macro_video_recall_iou50"],
                "macro_video_recall_ci95_low": bootstrap["bootstrap95_low"],
                "macro_video_recall_ci95_high": bootstrap["bootstrap95_high"],
                "center_precision_5px": metrics["top1"]["center_precision_5px"],
                "center_precision_10px": metrics["top1"]["center_precision_10px"],
                "center_precision_20px": metrics["top1"]["center_precision_20px"],
                "false_positive_frame_rate_absent": metrics["top1"]["false_positive_frame_rate_absent"],
                "false_positive_detections_per_absent_frame": metrics["top1"]["false_positive_detections_per_absent_frame"],
            }
        )
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(output_dir / "main_results.csv", index=False, encoding="utf-8-sig")

    pairs: list[dict[str, object]] = []
    for pair_index, (model_a, model_b) in enumerate(itertools.combinations(args.models, 2)):
        paired = pd.concat(
            [video_results[model_a].rename("a"), video_results[model_b].rename("b")], axis=1, join="inner"
        ).dropna()
        if len(paired) != 35:
            raise ValueError(f"Expected 35 paired videos for {model_a} vs {model_b}, found {len(paired)}")
        differences = paired["a"].to_numpy() - paired["b"].to_numpy()
        rng = np.random.default_rng(20260821 + pair_index)
        draws = rng.choice(differences, size=(args.resamples, len(differences)), replace=True).mean(axis=1)
        p_value = min(1.0, 2.0 * min(float((draws <= 0).mean()), float((draws >= 0).mean())))
        pairs.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "videos": len(differences),
                "mean_recall_difference_a_minus_b": float(differences.mean()),
                "bootstrap95_low": float(np.quantile(draws, 0.025)),
                "bootstrap95_high": float(np.quantile(draws, 0.975)),
                "two_sided_bootstrap_p": p_value,
                "resamples": args.resamples,
            }
        )
    adjusted = holm_adjust([float(row["two_sided_bootstrap_p"]) for row in pairs])
    for row, value in zip(pairs, adjusted):
        row["holm_adjusted_p"] = value
    pd.DataFrame(pairs).to_csv(output_dir / "paired_video_comparisons.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "protocol_version": "ard100-det-v1",
        "comparison_id": args.comparison_id,
        "models": args.models,
        "primary_paired_unit": "official test video",
        "paired_metric": "top-1 localization recall at IoU>=0.50",
        "bootstrap_resamples": args.resamples,
        "multiple_comparison_control": "Holm family-wise adjustment over all model pairs",
        "files": ["main_results.csv", "paired_video_comparisons.csv"],
    }
    (output_dir / "comparison_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
