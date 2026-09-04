#!/usr/bin/env python3
"""Clustered multivariable associations for locked IoU50 localization outcomes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "supplementary" / "multivariable_v3"
METADATA = ROOT / "datasets" / "ARD100_FullFrame_v1" / "metadata" / "test_frames.csv"
MODELS = [
    ("fasterrcnn_r50_fpn_v2", "Faster R-CNN"),
    ("retinanet_resnet50_fpn_v2", "RetinaNet"),
    ("fcos_resnet50_fpn", "FCOS"),
    ("yolov8s", "YOLOv8s"),
    ("rtdetr_l", "RT-DETR-L"),
    ("yolomg_paper", "YOLOMG-Arch"),
]
FEATURES = [
    ("log_size", "Target scale"),
    ("contrast", "Local contrast"),
    ("log_blur", "Laplacian variance"),
    ("clutter", "Edge-density clutter"),
    ("log_target_motion", "Target displacement"),
    ("log_camera_motion", "Camera displacement"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def design(metadata: pd.DataFrame) -> pd.DataFrame:
    frame = metadata[metadata["target_present"] == 1].copy()
    frame["log_size"] = np.log(frame["target_sqrt_area_px"].clip(lower=1e-6))
    frame["contrast"] = frame["target_contrast"]
    frame["log_blur"] = np.log1p(frame["target_context_laplacian_var"].clip(lower=0))
    frame["clutter"] = frame["context_edge_density"]
    frame["log_target_motion"] = np.log1p(frame["target_displacement_per_frame"].clip(lower=0))
    frame["log_camera_motion"] = np.log1p(frame["camera_displacement_px"].clip(lower=0))
    for feature, _label in FEATURES:
        values = frame[feature].replace([np.inf, -np.inf], np.nan)
        values = values.fillna(values.median())
        std = float(values.std(ddof=0))
        frame[feature] = (values - values.mean()) / (std if std > 0 else 1.0)
    return frame


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    base = design(pd.read_csv(METADATA))
    records = []
    hashes = {str(METADATA.relative_to(ROOT)): sha256(METADATA)}
    for model_id, display in MODELS:
        path = ROOT / "outputs" / "metrics" / model_id / "test_frame_metrics.csv"
        hashes[str(path.relative_to(ROOT))] = sha256(path)
        outcomes = pd.read_csv(path, usecols=["sample_id", "localized_iou50"])
        frame = base.merge(outcomes, on="sample_id", how="inner", validate="one_to_one")
        # Video fixed effects absorb stable between-video difficulty. Clustered
        # covariance then protects the feature intervals from treating adjacent
        # frames as independent. This formulation is numerically more stable
        # than estimating a working GEE correlation with only 35 clusters.
        video_effects = pd.get_dummies(frame["video_id"], prefix="video", drop_first=True, dtype=float)
        x = pd.concat([frame[[name for name, _label in FEATURES]].reset_index(drop=True), video_effects.reset_index(drop=True)], axis=1)
        x = sm.add_constant(x, has_constant="add")
        model = sm.GLM(frame["localized_iou50"].astype(float).reset_index(drop=True), x, family=sm.families.Binomial())
        fitted = model.fit(
            maxiter=200,
            cov_type="cluster",
            cov_kwds={"groups": frame["video_id"].reset_index(drop=True)},
        )
        for feature, label in FEATURES:
            beta = float(fitted.params[feature])
            se = float(fitted.bse[feature])
            records.append(
                {
                    "model": model_id,
                    "display": display,
                    "feature": feature,
                    "feature_label": label,
                    "standardization": "one dataset standard deviation",
                    "log_odds_coefficient": beta,
                    "robust_se": se,
                    "odds_ratio": float(np.exp(beta)),
                    "odds_ratio_ci95_low": float(np.exp(beta - 1.96 * se)),
                    "odds_ratio_ci95_high": float(np.exp(beta + 1.96 * se)),
                    "p_value": float(fitted.pvalues[feature]),
                    "frames": len(frame),
                    "clusters": int(frame["video_id"].nunique()),
                    "converged": bool(fitted.converged),
                }
            )
    results = pd.DataFrame(records)
    results.to_csv(OUT / "clustered_glm_associations.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.1), sharex=True)
    for ax, (model_id, display) in zip(axes.ravel(), MODELS):
        part = results[results["model"] == model_id].iloc[::-1]
        y = np.arange(len(part))
        ax.errorbar(
            part["odds_ratio"], y,
            xerr=[part["odds_ratio"] - part["odds_ratio_ci95_low"], part["odds_ratio_ci95_high"] - part["odds_ratio"]],
            fmt="o", capsize=2,
        )
        ax.axvline(1.0, color="0.4", linewidth=0.8, linestyle="--")
        ax.set_yticks(y, part["feature_label"], fontsize=7)
        ax.set_xscale("log")
        ax.set_title(display, fontsize=9)
        ax.grid(axis="x", alpha=0.2)
    fig.supxlabel("Odds ratio for IoU50 localization per 1-SD attribute increase")
    fig.tight_layout()
    fig.savefig(OUT / "clustered_glm_forest.png", dpi=240)
    fig.savefig(OUT / "clustered_glm_forest.pdf")
    plt.close(fig)
    manifest = {
        "protocol": "ard100-det-supp-v2",
        "status": "PASS" if results["converged"].all() else "WARN",
        "model": "binomial GLM with video fixed effects and video-clustered covariance",
        "interpretation": "adjusted association, not causal effect",
        "input_hashes": hashes,
        "outputs": ["clustered_glm_associations.csv", "clustered_glm_forest.png", "clustered_glm_forest.pdf"],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
