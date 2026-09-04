#!/usr/bin/env python3
"""Summarize ARD100 test factors and create the dataset-characteristics figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "datasets" / "ARD100_FullFrame_v1" / "metadata" / "test_frames.csv"
OUTPUT = ROOT / "outputs" / "dataset_analysis"


def percentiles(series: pd.Series) -> dict[str, float]:
    clean = series[np.isfinite(series)].astype(float)
    return {
        "count": int(len(clean)),
        "p05": float(clean.quantile(0.05)),
        "p10": float(clean.quantile(0.10)),
        "p25": float(clean.quantile(0.25)),
        "median": float(clean.quantile(0.50)),
        "p75": float(clean.quantile(0.75)),
        "p90": float(clean.quantile(0.90)),
        "p95": float(clean.quantile(0.95)),
    }


def ecdf(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values[np.isfinite(values)].to_numpy(dtype=float))
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(METADATA)
    positive = data[data["target_present"] == 1].copy()
    factors = [
        "target_width_px",
        "target_height_px",
        "target_sqrt_area_px",
        "target_contrast",
        "target_context_laplacian_var",
        "context_edge_density",
        "target_displacement_per_frame",
        "camera_displacement_px",
        "phase_response",
    ]
    summary = {
        "protocol_version": "ard100-det-v1",
        "videos": int(data["video_id"].nunique()),
        "annotated_frames": int(len(data)),
        "target_present_frames": int(data["target_present"].sum()),
        "target_absent_frames": int((data["target_present"] == 0).sum()),
        "both_dimensions_le_20px_rate": float(
            ((positive["target_width_px"] <= 20) & (positive["target_height_px"] <= 20)).mean()
        ),
        "factors": {factor: percentiles(positive[factor]) for factor in factors},
    }
    (OUTPUT / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    per_video = data.groupby("video_id", sort=True).agg(
        annotated_frames=("sample_id", "count"),
        positive_frames=("target_present", "sum"),
        target_width_median_px=("target_width_px", "median"),
        target_height_median_px=("target_height_px", "median"),
        sqrt_area_median_px=("target_sqrt_area_px", "median"),
        contrast_median=("target_contrast", "median"),
        camera_displacement_median_px=("camera_displacement_px", "median"),
    ).reset_index()
    per_video["negative_frames"] = per_video["annotated_frames"] - per_video["positive_frames"]
    per_video.to_csv(OUTPUT / "per_video_dataset_summary.csv", index=False, encoding="utf-8-sig")

    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), constrained_layout=True)
    color = "#2166ac"

    x, y = ecdf(positive["target_sqrt_area_px"])
    axes[0, 0].plot(x, y, color=color, linewidth=1.6)
    for value in (12, 20, 32):
        axes[0, 0].axvline(value, color="#b2182b", linewidth=0.8, linestyle="--")
    axes[0, 0].set(xlabel="Target sqrt-area (original px)", ylabel="Empirical CDF", title="(a) Target scale")
    axes[0, 0].grid(alpha=0.2)

    axes[0, 1].hist(positive["target_contrast"].dropna(), bins=40, color=color, alpha=0.85)
    axes[0, 1].set(xlabel="Normalized target-ring contrast", ylabel="Frames", title="(b) Local contrast")

    blur = positive["target_context_laplacian_var"].clip(lower=1e-3)
    axes[0, 2].hist(np.log10(blur.dropna()), bins=40, color=color, alpha=0.85)
    axes[0, 2].set(xlabel="log10 Laplacian variance", ylabel="Frames", title="(c) Blur proxy")

    axes[1, 0].hist(positive["context_edge_density"].dropna(), bins=40, color=color, alpha=0.85)
    axes[1, 0].set(xlabel="Ring edge density", ylabel="Frames", title="(d) Clutter proxy")

    target_motion = positive["target_displacement_per_frame"].clip(lower=1e-3)
    axes[1, 1].hist(np.log10(target_motion.dropna()), bins=40, color=color, alpha=0.85)
    axes[1, 1].set(xlabel="log10 target displacement (px/frame)", ylabel="Frames", title="(e) Apparent target motion")

    camera_motion = positive["camera_displacement_px"].clip(lower=1e-3)
    axes[1, 2].hist(np.log10(camera_motion.dropna()), bins=40, color=color, alpha=0.85)
    axes[1, 2].set(xlabel="log10 camera displacement (px)", ylabel="Frames", title="(f) Camera-motion proxy")

    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT / "dataset_characteristics.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT / "dataset_characteristics.pdf", bbox_inches="tight")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

