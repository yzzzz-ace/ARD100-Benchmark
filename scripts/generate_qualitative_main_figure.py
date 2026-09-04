#!/usr/bin/env python3
"""Create a compact two-case qualitative figure for the main manuscript."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from generate_qualitative_figure import (
    COLORS,
    IMAGE_ROOT,
    METADATA,
    MODELS,
    ROOT,
    add_box,
    crop_bounds,
    read_selected_detections,
    select_cases,
    sha256,
)


OUT = ROOT / "outputs" / "supplementary" / "qualitative_main_v1"


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    metadata = pd.read_csv(METADATA)
    model_frames = {
        model_id: pd.read_csv(ROOT / "outputs" / "metrics" / model_id / "test_frame_metrics.csv")
        for model_id, _display in MODELS
    }
    all_cases = select_cases(metadata, model_frames)
    cases = [all_cases[0], all_cases[3]]
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
    metadata_index = metadata.set_index("sample_id")
    fig, axes = plt.subplots(2, 2, figsize=(3.45, 4.25), gridspec_kw={"width_ratios": [1.15, 1]})
    records = []
    for row_index, (case_label, sample_id) in enumerate(cases):
        row = metadata_index.loc[sample_id]
        bgr = cv2.imread(str(IMAGE_ROOT / f"{sample_id}.jpg"), cv2.IMREAD_COLOR)
        if bgr is None:
            raise OSError(IMAGE_ROOT / f"{sample_id}.jpg")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        case_detections = {model_id: detections[model_id][sample_id] for model_id, _display in MODELS}
        full_ax, crop_ax = axes[row_index]
        full_ax.imshow(rgb)
        crop_ax.imshow(rgb)
        if int(row.target_present):
            scale = float(row.stored_scale)
            gt = np.asarray([row.target_xmin, row.target_ymin, row.target_xmax, row.target_ymax], dtype=float) * scale
            add_box(full_ax, gt, "white", linewidth=1.6, linestyle="--")
            add_box(crop_ax, gt, "white", linewidth=1.6, linestyle="--")
        responses = []
        for (model_id, display), color in zip(MODELS, COLORS):
            if case_detections[model_id]:
                score, box = case_detections[model_id][0]
                add_box(full_ax, box, color, linewidth=1.0)
                add_box(crop_ax, box, color, linewidth=1.3)
                responses.append(f"{display}:{score:.2f}")
        x1, y1, x2, y2 = crop_bounds(row, case_detections, width, height)
        crop_ax.set_xlim(x1, x2)
        crop_ax.set_ylim(y2, y1)
        full_ax.set_title(f"({chr(97 + row_index)}) {case_label}\n{sample_id}", fontsize=6.6, loc="left")
        crop_ax.set_title("Magnified", fontsize=6.6)
        for ax in (full_ax, crop_ax):
            ax.set_xticks([])
            ax.set_yticks([])
        records.append({"case": case_label, "sample_id": sample_id, "above_threshold_models": responses})
    handles = [Rectangle((0, 0), 1, 1, fill=False, edgecolor="black", linewidth=1.4, linestyle="--", label="GT")]
    handles += [
        Rectangle((0, 0), 1, 1, fill=False, edgecolor=color, linewidth=1.4, label=display)
        for (_model, display), color in zip(MODELS, COLORS)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=5.5, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1), pad=0.5)
    fig.savefig(OUT / "qualitative_main.png", dpi=300)
    fig.savefig(OUT / "qualitative_main.pdf")
    plt.close(fig)
    manifest = {
        "protocol": "ard100-det-supp-v2",
        "status": "PASS",
        "selection": "two main-paper cases inherited from the four-case deterministic qualitative manifest",
        "cases": records,
        "metadata_sha256": sha256(METADATA),
        "outputs": ["qualitative_main.png", "qualitative_main.pdf"],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
