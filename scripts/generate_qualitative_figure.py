#!/usr/bin/env python3
"""Generate deterministic qualitative panels from locked predictions."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "supplementary" / "qualitative_v2"
METADATA = ROOT / "datasets" / "ARD100_FullFrame_v1" / "metadata" / "test_frames.csv"
IMAGE_ROOT = ROOT / "datasets" / "ARD100_FullFrame_v1" / "images" / "test"
MODELS = [
    ("fasterrcnn_r50_fpn_v2", "Faster R-CNN"),
    ("retinanet_resnet50_fpn_v2", "RetinaNet"),
    ("fcos_resnet50_fpn", "FCOS"),
    ("yolov8s", "YOLOv8s"),
    ("rtdetr_l", "RT-DETR-L"),
    ("yolomg_paper", "YOLOMG-Arch"),
]
COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#00bfc4"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_cases(metadata: pd.DataFrame, model_frames: dict[str, pd.DataFrame]) -> list[tuple[str, str]]:
    joined = metadata.copy()
    localized_columns = []
    prediction_columns = []
    for model_id, _display in MODELS:
        part = model_frames[model_id][["sample_id", "localized_iou50", "prediction_present"]].rename(
            columns={
                "localized_iou50": f"loc_{model_id}",
                "prediction_present": f"pred_{model_id}",
            }
        )
        joined = joined.merge(part, on="sample_id", how="left", validate="one_to_one")
        localized_columns.append(f"loc_{model_id}")
        prediction_columns.append(f"pred_{model_id}")
    joined["localized_count"] = joined[localized_columns].sum(axis=1)
    joined["prediction_count"] = joined[prediction_columns].sum(axis=1)
    joined["disagreement"] = 3 - (joined["localized_count"] - 3).abs()
    positive = joined[joined["target_present"] == 1].copy()
    contrast_q = float(positive["target_contrast"].quantile(1 / 3))
    camera_q = float(positive["camera_displacement_px"].quantile(0.90))
    rules = [
        ("Tiny and low contrast", positive[(positive["target_sqrt_area_px"] <= 12) & (positive["target_contrast"] <= contrast_q)]),
        ("Large camera motion", positive[positive["camera_displacement_px"] >= camera_q]),
        ("Near-stationary target", positive[positive["target_displacement_per_frame"] <= 0.5]),
    ]
    chosen = []
    for label, eligible in rules:
        selected = eligible.sort_values(["disagreement", "localized_count", "sample_id"], ascending=[False, False, True]).iloc[0]
        chosen.append((label, str(selected["sample_id"])))
    absent = joined[joined["target_present"] == 0].sort_values(
        ["prediction_count", "sample_id"], ascending=[False, True]
    ).iloc[0]
    chosen.append(("Explicit target-absent frame", str(absent["sample_id"])))
    return chosen


def read_selected_detections(path: Path, sample_ids: set[str], thresholds: dict[str, float], model_id: str):
    output = {sample_id: [] for sample_id in sample_ids}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            sample_id = row["sample_id"]
            if sample_id not in sample_ids or float(row["score"]) < thresholds[model_id]:
                continue
            output[sample_id].append(
                (
                    float(row["score"]),
                    np.asarray([float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])], dtype=float),
                )
            )
    for sample_id in output:
        output[sample_id].sort(key=lambda item: item[0], reverse=True)
        output[sample_id] = output[sample_id][:1]
    return output


def add_box(ax, box, color, linewidth=1.6, linestyle="-"):
    x1, y1, x2, y2 = box
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=linewidth, linestyle=linestyle))


def crop_bounds(row, detections, width, height):
    boxes = []
    if int(row.target_present):
        scale = float(row.stored_scale)
        gt = np.asarray([row.target_xmin, row.target_ymin, row.target_xmax, row.target_ymax], dtype=float) * scale
        # Positive-frame inset is centered only on the annotation. Distant false
        # positives remain visible in the full-frame panel and must not dilute
        # the promised magnification of a few-pixel target.
        boxes.append(gt)
    else:
        for model_values in detections.values():
            if model_values:
                boxes.append(model_values[0][1])
    if not boxes:
        return 0, 0, width, height
    stack = np.vstack(boxes)
    center = np.asarray([(stack[:, [0, 2]].min() + stack[:, [0, 2]].max()) / 2, (stack[:, [1, 3]].min() + stack[:, [1, 3]].max()) / 2]).ravel()
    side = max(120.0, float(stack[:, 2].max() - stack[:, 0].min()) * 4, float(stack[:, 3].max() - stack[:, 1].min()) * 4)
    x1 = max(0, center[0] - side / 2)
    y1 = max(0, center[1] - side / 2)
    x2 = min(width, x1 + side)
    y2 = min(height, y1 + side)
    x1 = max(0, x2 - side)
    y1 = max(0, y2 - side)
    return x1, y1, x2, y2


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    metadata = pd.read_csv(METADATA)
    model_frames = {
        model_id: pd.read_csv(ROOT / "outputs" / "metrics" / model_id / "test_frame_metrics.csv")
        for model_id, _display in MODELS
    }
    cases = select_cases(metadata, model_frames)
    sample_ids = {sample_id for _label, sample_id in cases}
    thresholds = {
        model_id: float(json.loads((ROOT / "outputs" / "metrics" / model_id / "threshold.json").read_text())["threshold"])
        for model_id, _display in MODELS
    }
    detections = {
        model_id: read_selected_detections(
            ROOT / "outputs" / "predictions" / model_id / "test.csv", sample_ids, thresholds, model_id
        )
        for model_id, _display in MODELS
    }

    fig, axes = plt.subplots(4, 2, figsize=(7.2, 9.0), gridspec_kw={"width_ratios": [1.45, 1]})
    selected_records = []
    metadata_index = metadata.set_index("sample_id")
    for row_index, (case_label, sample_id) in enumerate(cases):
        row = metadata_index.loc[sample_id]
        image_path = IMAGE_ROOT / f"{sample_id}.jpg"
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise OSError(image_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        case_detections = {model_id: detections[model_id][sample_id] for model_id, _display in MODELS}
        full_ax, crop_ax = axes[row_index]
        full_ax.imshow(rgb)
        crop_ax.imshow(rgb)
        gt_box = None
        if int(row.target_present):
            scale = float(row.stored_scale)
            gt_box = np.asarray([row.target_xmin, row.target_ymin, row.target_xmax, row.target_ymax], dtype=float) * scale
            add_box(full_ax, gt_box, "white", linewidth=2.2, linestyle="--")
            add_box(crop_ax, gt_box, "white", linewidth=2.2, linestyle="--")
        response_models = []
        for (model_id, display), color in zip(MODELS, COLORS):
            if case_detections[model_id]:
                score, box = case_detections[model_id][0]
                add_box(full_ax, box, color)
                add_box(crop_ax, box, color, linewidth=2.0)
                response_models.append(f"{display}:{score:.2f}")
        x1, y1, x2, y2 = crop_bounds(row, case_detections, width, height)
        crop_ax.set_xlim(x1, x2)
        crop_ax.set_ylim(y2, y1)
        full_ax.set_title(f"({chr(97 + row_index)}) {case_label}: {sample_id}", fontsize=8, loc="left")
        crop_ax.set_title("Magnified region", fontsize=8)
        for ax in (full_ax, crop_ax):
            ax.set_xticks([])
            ax.set_yticks([])
        selected_records.append(
            {
                "case": case_label,
                "sample_id": sample_id,
                "target_present": int(row.target_present),
                "target_sqrt_area_px": float(row.target_sqrt_area_px) if int(row.target_present) else None,
                "target_contrast": float(row.target_contrast) if int(row.target_present) else None,
                "target_displacement_per_frame": float(row.target_displacement_per_frame) if int(row.target_present) else None,
                "camera_displacement_px": float(row.camera_displacement_px) if int(row.target_present) else None,
                "above_threshold_models": response_models,
            }
        )
    handles = [Rectangle((0, 0), 1, 1, fill=False, edgecolor="white", linewidth=2, linestyle="--", label="Ground truth")]
    handles += [Rectangle((0, 0), 1, 1, fill=False, edgecolor=color, linewidth=2, label=display) for (_model, display), color in zip(MODELS, COLORS)]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(OUT / "qualitative_cases.png", dpi=240)
    fig.savefig(OUT / "qualitative_cases.pdf")
    plt.close(fig)
    manifest = {
        "protocol": "ard100-det-supp-v2",
        "status": "PASS",
        "selection_rule": "maximum six-model disagreement within predeclared regimes; absent frame uses maximum locked response count; lexicographic tie break",
        "cases": selected_records,
        "metadata_sha256": sha256(METADATA),
        "outputs": ["qualitative_cases.png", "qualitative_cases.pdf"],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
