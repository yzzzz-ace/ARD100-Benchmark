#!/usr/bin/env python3
"""Factorize locked 640-input localization recall with video-level uncertainty."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "datasets" / "ARD100_FullFrame_v1" / "metadata" / "test_frames.csv"
LOCKED_TEST = ROOT / "outputs" / "locked_test_manifest.json"
OUTPUT = ROOT / "outputs" / "factor_analysis"
MODELS = (
    "fasterrcnn_r50_fpn_v2",
    "retinanet_resnet50_fpn_v2",
    "fcos_resnet50_fpn",
    "yolov8s",
    "rtdetr_l",
    "yolomg_paper",
)
DISPLAY = {
    "fasterrcnn_r50_fpn_v2": "Faster R-CNN",
    "retinanet_resnet50_fpn_v2": "RetinaNet",
    "fcos_resnet50_fpn": "FCOS",
    "yolov8s": "YOLOv8s",
    "rtdetr_l": "RT-DETR-L",
    "yolomg_paper": "YOLOMG-Arch",
}
FACTORS = {
    "target_sqrt_area_px": {
        "edges": [-np.inf, 12.0, 20.0, 32.0, np.inf],
        "labels": ["<=12", "12-20", "20-32", ">32"],
        "meaning": "original-resolution square-root box area (px)",
    },
    "target_contrast": {
        "edges": [-np.inf, 0.05, 0.10, np.inf],
        "labels": ["<=0.05", "0.05-0.10", ">0.10"],
        "meaning": "absolute target/ring grayscale contrast normalized by 255",
    },
    "target_context_laplacian_var": {
        "edges": [-np.inf, 350.0, 1400.0, np.inf],
        "labels": ["<=350", "350-1400", ">1400"],
        "meaning": "local Laplacian variance (low values indicate blur or weak texture)",
    },
    "context_edge_density": {
        "edges": [-np.inf, 0.06, 0.22, np.inf],
        "labels": ["<=0.06", "0.06-0.22", ">0.22"],
        "meaning": "surrounding-ring Canny edge density",
    },
    "target_displacement_per_frame": {
        "edges": [-np.inf, 1.0, 4.0, np.inf],
        "labels": ["<=1", "1-4", ">4"],
        "meaning": "target-center displacement per annotated frame (px)",
    },
    "camera_displacement_px": {
        "edges": [-np.inf, 0.5, 3.0, np.inf],
        "labels": ["<=0.5", "0.5-3", ">3"],
        "meaning": "phase-correlation camera displacement magnitude (px)",
    },
}
RESAMPLES = 10000
SEED = 20260821


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def video_bootstrap_recall(frame: pd.DataFrame, seed: int) -> tuple[float, float]:
    per_video = frame.groupby("video_id", sort=True)["localized_iou50"].agg(["sum", "count"])
    if per_video.empty:
        return float("nan"), float("nan")
    successes = per_video["sum"].to_numpy(dtype=np.float64)
    totals = per_video["count"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws: list[np.ndarray] = []
    remaining = RESAMPLES
    while remaining:
        chunk = min(500, remaining)
        indices = rng.integers(0, len(per_video), size=(chunk, len(per_video)))
        numerator = successes[indices].sum(axis=1)
        denominator = totals[indices].sum(axis=1)
        draws.append(numerator / np.maximum(denominator, 1.0))
        remaining -= chunk
    values = np.concatenate(draws)
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def main() -> None:
    if not LOCKED_TEST.exists():
        raise RuntimeError("Factor analysis is locked until outputs/locked_test_manifest.json exists")
    locked = json.loads(LOCKED_TEST.read_text(encoding="utf-8"))
    completed_ids = {entry["output_id"] for entry in locked["entries"]}
    missing_models = sorted(set(MODELS) - completed_ids)
    if missing_models:
        raise RuntimeError(f"Locked 640 results are incomplete: {missing_models}")

    metadata = pd.read_csv(METADATA)
    metadata = metadata[metadata["target_present"].astype(bool)].copy()
    if len(metadata) != 71633 or metadata["video_id"].nunique() != 35:
        raise ValueError("Unexpected target-present test denominator")

    rows: list[dict[str, object]] = []
    for model_index, model_id in enumerate(MODELS):
        metric_path = ROOT / "outputs" / "metrics" / model_id / "test_frame_metrics.csv"
        metrics = pd.read_csv(metric_path)
        merged = metadata.merge(
            metrics[
                [
                    "sample_id",
                    "localized_iou50",
                    "prediction_present",
                    "top1_iou",
                    "center_error_original_px",
                ]
            ],
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
        if merged["localized_iou50"].isna().any():
            raise ValueError(f"Missing frame metric after merge for {model_id}")
        for factor_index, (factor, spec) in enumerate(FACTORS.items()):
            binned = pd.cut(
                merged[factor],
                bins=spec["edges"],
                labels=spec["labels"],
                right=True,
                include_lowest=True,
            )
            for bin_index, label in enumerate(spec["labels"]):
                part = merged[binned == label].copy()
                if part.empty:
                    continue
                recall = float(part["localized_iou50"].mean())
                center10 = float(np.isfinite(part["center_error_original_px"]).mul(
                    part["center_error_original_px"] <= 10.0
                ).mean())
                low, high = video_bootstrap_recall(
                    part,
                    SEED + model_index * 100 + factor_index * 10 + bin_index,
                )
                rows.append(
                    {
                        "model": model_id,
                        "model_display": DISPLAY[model_id],
                        "factor": factor,
                        "factor_meaning": spec["meaning"],
                        "bin": label,
                        "positive_frames": len(part),
                        "videos_represented": int(part["video_id"].nunique()),
                        "top1_recall_iou50": recall,
                        "video_cluster_bootstrap95_low": low,
                        "video_cluster_bootstrap95_high": high,
                        "center_precision_10px": center10,
                        "prediction_present_rate": float(part["prediction_present"].mean()),
                        "mean_top1_iou": float(part["top1_iou"].mean()),
                    }
                )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT / "factor_results.csv", index=False, encoding="utf-8-sig")

    ordered_rows = [
        f"{factor}: {label}"
        for factor, spec in FACTORS.items()
        for label in spec["labels"]
    ]
    plot_frame = result.copy()
    plot_frame["factor_bin"] = plot_frame["factor"] + ": " + plot_frame["bin"].astype(str)
    heatmap = plot_frame.pivot(index="factor_bin", columns="model_display", values="top1_recall_iou50")
    heatmap = heatmap.reindex(index=ordered_rows, columns=[DISPLAY[model] for model in MODELS])
    sns.set_theme(style="white", font_scale=0.86)
    figure, axis = plt.subplots(figsize=(10.8, 8.0))
    sns.heatmap(
        heatmap,
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Top-1 localization recall (IoU >= 0.50)"},
        ax=axis,
    )
    axis.set_xlabel("")
    axis.set_ylabel("Predeclared factor stratum")
    figure.tight_layout()
    figure.savefig(OUTPUT / "factor_recall_heatmap.png", dpi=300, bbox_inches="tight")
    figure.savefig(OUTPUT / "factor_recall_heatmap.pdf", bbox_inches="tight")
    plt.close(figure)

    manifest = {
        "protocol_version": "ard100-det-v1",
        "source_locked_test_manifest": str(LOCKED_TEST.resolve()),
        "source_locked_test_manifest_sha256": sha256(LOCKED_TEST),
        "input_resolution": 640,
        "models": list(MODELS),
        "positive_frame_denominator": 71633,
        "bootstrap_unit": "official test video",
        "bootstrap_resamples": RESAMPLES,
        "factor_thresholds_predeclared_before_test_predictions": {
            factor: {
                "edges": [
                    "-inf" if np.isneginf(value) else "+inf" if np.isposinf(value) else float(value)
                    for value in spec["edges"]
                ],
                "labels": spec["labels"],
                "meaning": spec["meaning"],
            }
            for factor, spec in FACTORS.items()
        },
        "files": ["factor_results.csv", "factor_recall_heatmap.png", "factor_recall_heatmap.pdf"],
        "interpretation": "descriptive conditional performance; not a causal effect estimate",
    }
    (OUTPUT / "factor_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(result), "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
