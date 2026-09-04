#!/usr/bin/env python3
"""Generate manuscript figures strictly from the locked ARD100 outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper" / "latex" / "figures"
LOCKED_TEST = ROOT / "outputs" / "locked_test_manifest.json"

ORDER = [
    "fasterrcnn_r50_fpn_v2",
    "retinanet_resnet50_fpn_v2",
    "fcos_resnet50_fpn",
    "yolov8s",
    "rtdetr_l",
    "yolomg_paper",
]
DISPLAY = {
    "fasterrcnn_r50_fpn_v2": "Faster R-CNN",
    "retinanet_resnet50_fpn_v2": "RetinaNet",
    "fcos_resnet50_fpn": "FCOS",
    "yolov8s": "YOLOv8s",
    "rtdetr_l": "RT-DETR-L",
    "yolomg_paper": "YOLOMG-Arch",
}
COLORS = {
    "fasterrcnn_r50_fpn_v2": "#4C78A8",
    "retinanet_resnet50_fpn_v2": "#F58518",
    "fcos_resnet50_fpn": "#E45756",
    "yolov8s": "#72B7B2",
    "rtdetr_l": "#54A24B",
    "yolomg_paper": "#B279A2",
}


def require_locked_outputs() -> None:
    if not LOCKED_TEST.exists():
        raise RuntimeError("Paper figures require outputs/locked_test_manifest.json")
    manifest = json.loads(LOCKED_TEST.read_text(encoding="utf-8-sig"))
    entries = manifest.get("entries", [])
    if len(entries) != 12 or any(int(entry.get("frames", -1)) != 72631 for entry in entries):
        raise RuntimeError("Locked test manifest does not contain twelve complete 72,631-frame entries")


def runtime_fps(model_id: str, image_size: int) -> float:
    path = ROOT / "outputs" / "runtime" / model_id / f"i{image_size}_fp32.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["throughput_fps_from_mean"])


def main() -> None:
    require_locked_outputs()
    main_640 = pd.read_csv(ROOT / "outputs" / "comparisons" / "main_i640" / "main_results.csv")
    main_1280 = pd.read_csv(
        ROOT / "outputs" / "comparisons" / "resolution_i1280" / "main_results.csv"
    )
    factors = pd.read_csv(ROOT / "outputs" / "factor_analysis" / "factor_results.csv")

    table_640 = main_640.set_index("model")
    main_1280 = main_1280.assign(base_model=main_1280["model"].str.replace("_i1280$", "", regex=True))
    table_1280 = main_1280.set_index("base_model")
    if list(table_640.loc[ORDER].index) != ORDER or list(table_1280.loc[ORDER].index) != ORDER:
        raise RuntimeError("Unexpected model identity or order in locked comparison outputs")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.3), constrained_layout=True)

    # (a) Accuracy--throughput operating points at the primary 640 resolution.
    ax = axes[0]
    for model_id in ORDER:
        x = runtime_fps(model_id, 640)
        y = float(table_640.loc[model_id, "AP50_95"])
        ax.scatter(x, y, s=65, color=COLORS[model_id], edgecolor="white", linewidth=0.7, zorder=3)
        offset = (4, 4)
        if model_id == "retinanet_resnet50_fpn_v2":
            offset = (4, -11)
        elif model_id == "fcos_resnet50_fpn":
            offset = (4, -1)
        ax.annotate(DISPLAY[model_id], (x, y), xytext=offset, textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Model throughput (frames/s, log scale)")
    ax.set_ylabel(r"Official-test $\mathrm{AP}_{50:95}$")
    ax.set_title("(a) Accuracy--throughput at 640")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
    ax.set_ylim(0.0, 0.35)

    # (b) Paired checkpoint sensitivity to inference resolution.
    ax = axes[1]
    y_pos = np.arange(len(ORDER))
    for index, model_id in enumerate(ORDER):
        ap_640 = float(table_640.loc[model_id, "AP50_95"])
        ap_1280 = float(table_1280.loc[model_id, "AP50_95"])
        ax.plot([ap_640, ap_1280], [index, index], color="#B0B0B0", linewidth=2, zorder=1)
        ax.scatter(ap_640, index, marker="o", s=45, color=COLORS[model_id], zorder=2)
        ax.scatter(ap_1280, index, marker="D", s=43, color=COLORS[model_id], zorder=2)
        ax.text(max(ap_640, ap_1280) + 0.006, index, f"{ap_1280 - ap_640:+.3f}", va="center", fontsize=7.5)
    ax.set_yticks(y_pos, [DISPLAY[model_id] for model_id in ORDER])
    ax.invert_yaxis()
    ax.set_xlabel(r"Official-test $\mathrm{AP}_{50:95}$")
    ax.set_title("(b) 640 to 1280 without retraining")
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.6)
    ax.scatter([], [], marker="o", color="#555555", label="640")
    ax.scatter([], [], marker="D", color="#555555", label="1280")
    ax.legend(loc="upper right", frameon=False)
    ax.set_xlim(0.0, 0.35)

    # (c) The predeclared target-scale factor at the primary resolution.
    ax = axes[2]
    size = factors[factors["factor"] == "target_sqrt_area_px"].copy()
    size["model"] = pd.Categorical(size["model"], ORDER, ordered=True)
    size["bin"] = pd.Categorical(size["bin"], ["<=12", "12-20", "20-32", ">32"], ordered=True)
    heat = size.pivot(index="model", columns="bin", values="top1_recall_iou50").loc[ORDER]
    image = ax.imshow(heat.to_numpy(), cmap="viridis", vmin=0.0, vmax=0.9, aspect="auto")
    ax.set_xticks(np.arange(4), [r"$\leq12$", "12--20", "20--32", r"$>32$"])
    ax.set_yticks(np.arange(len(ORDER)), [DISPLAY[model_id] for model_id in ORDER])
    ax.set_xlabel("Original-resolution square-root area (px)")
    ax.set_title("(c) Conditional IoU50 recall by scale")
    for row in range(heat.shape[0]):
        for col in range(heat.shape[1]):
            value = float(heat.iloc[row, col])
            color = "white" if value < 0.30 or value > 0.72 else "black"
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=color, fontsize=7.5)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("Recall")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pdf = OUTPUT / "locked_benchmark_summary.pdf"
    png = OUTPUT / "locked_benchmark_summary.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    print(json.dumps({"pdf": str(pdf), "png": str(png)}, indent=2))


if __name__ == "__main__":
    main()
