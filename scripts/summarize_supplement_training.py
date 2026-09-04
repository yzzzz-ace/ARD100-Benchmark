#!/usr/bin/env python3
"""Summarize three-seed validation stability and the leakage diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "supplementary" / "training_summary"
PRIMARY_IDS = {
    "yolov8s": "yolov8s",
    "rtdetr_l": "rtdetr_l",
    "yolomg_arch": "yolomg_paper",
}
SUPPLEMENT_IDS = {
    "yolov8s": ["yolov8s_video_disjoint_seed20260822", "yolov8s_video_disjoint_seed20260823"],
    "rtdetr_l": ["rtdetr_l_video_disjoint_seed20260822", "rtdetr_l_video_disjoint_seed20260823"],
    "yolomg_arch": ["yolomg_arch_video_disjoint_seed2", "yolomg_arch_video_disjoint_seed3"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def common_metrics(output_id: str) -> tuple[float, float, Path]:
    path = ROOT / "outputs" / "metrics" / output_id / "val_metrics.json"
    payload = json.loads(path.read_text())
    return (
        float(payload["average_precision"]["AP50_95"]),
        float(payload["top1"]["macro_video_recall_iou50"]),
        path,
    )


def best_native_map(results_path: Path, field: str) -> float:
    frame = pd.read_csv(results_path)
    frame.columns = [column.strip() for column in frame.columns]
    return float(frame[field].max())


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    rows = []
    hashes = {}
    seed_labels = {
        "yolov8s": ["20260821", "20260822", "20260823"],
        "rtdetr_l": ["20260821", "20260822", "20260823"],
        "yolomg_arch": ["1", "2", "3"],
    }
    for model, primary_id in PRIMARY_IDS.items():
        output_ids = [primary_id] + SUPPLEMENT_IDS[model]
        for seed, output_id in zip(seed_labels[model], output_ids):
            ap, macro, path = common_metrics(output_id)
            hashes[str(path.relative_to(ROOT))] = sha256(path)
            rows.append({"model": model, "seed": seed, "output_id": output_id, "val_AP50_95": ap, "val_macro_recall": macro})
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "seed_runs.csv", index=False, encoding="utf-8-sig")
    summary = runs.groupby("model").agg(
        seeds=("seed", "count"),
        val_AP50_95_mean=("val_AP50_95", "mean"),
        val_AP50_95_sd=("val_AP50_95", "std"),
        val_AP50_95_min=("val_AP50_95", "min"),
        val_AP50_95_max=("val_AP50_95", "max"),
        val_macro_recall_mean=("val_macro_recall", "mean"),
        val_macro_recall_sd=("val_macro_recall", "std"),
        val_macro_recall_min=("val_macro_recall", "min"),
        val_macro_recall_max=("val_macro_recall", "max"),
    ).reset_index()
    summary.to_csv(OUT / "seed_summary.csv", index=False, encoding="utf-8-sig")

    current_results = ROOT / "outputs" / "training" / "ultralytics" / "yolov8s_ard100_det_v1_nativeaug_sgd_e30_b64" / "results.csv"
    random_results = ROOT / "outputs" / "supplementary" / "training" / "yolov8s_random_frame_seed20260821" / "results.csv"
    for path in (current_results, random_results):
        hashes[str(path.relative_to(ROOT))] = sha256(path)
    video_disjoint = best_native_map(current_results, "metrics/mAP50-95(B)")
    random_frame = best_native_map(random_results, "metrics/mAP50-95(B)")
    leakage = {
        "detector": "yolov8s",
        "metric": "native validation AP50:95",
        "video_disjoint": video_disjoint,
        "random_frame": random_frame,
        "absolute_inflation": random_frame - video_disjoint,
        "relative_inflation": (random_frame - video_disjoint) / max(video_disjoint, 1e-12),
    }
    (OUT / "leakage_diagnostic.json").write_text(json.dumps(leakage, indent=2) + "\n")
    manifest = {
        "protocol": "ard100-det-supp-v2",
        "status": "PASS",
        "official_test_reused": False,
        "input_hashes": hashes,
        "outputs": ["seed_runs.csv", "seed_summary.csv", "leakage_diagnostic.json"],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": manifest, "leakage": leakage}, indent=2))


if __name__ == "__main__":
    main()
