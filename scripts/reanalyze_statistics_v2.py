#!/usr/bin/env python3
"""Correct paired video bootstrap inference without replacing locked tables."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "supplementary" / "statistics_v2"
MODELS = [
    "fasterrcnn_r50_fpn_v2",
    "retinanet_resnet50_fpn_v2",
    "fcos_resnet50_fpn",
    "yolov8s",
    "rtdetr_l",
    "yolomg_paper",
]
RESAMPLES = 10000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def per_video(frame: pd.DataFrame) -> pd.Series:
    return frame.groupby("video_id", sort=True).apply(
        lambda part: part["localized_iou50"].sum() / max(part["target_present"].sum(), 1),
        include_groups=False,
    )


def holm(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p_values[int(index)])
        adjusted[int(index)] = min(running, 1.0)
    return adjusted.tolist()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    values = {}
    hashes = {}
    for model in MODELS:
        path = ROOT / "outputs" / "metrics" / model / "test_frame_metrics.csv"
        hashes[str(path.relative_to(ROOT))] = sha256(path)
        values[model] = per_video(pd.read_csv(path))

    rows = []
    for pair_index, (model_a, model_b) in enumerate(itertools.combinations(MODELS, 2)):
        paired = pd.concat([values[model_a].rename("a"), values[model_b].rename("b")], axis=1).dropna()
        if len(paired) != 35:
            raise ValueError(f"Expected 35 paired videos, got {len(paired)}")
        differences = (paired["a"] - paired["b"]).to_numpy(dtype=float)
        rng = np.random.default_rng(20260821 + pair_index)
        draws = rng.choice(differences, size=(RESAMPLES, len(differences)), replace=True).mean(axis=1)
        lower_count = int((draws <= 0).sum())
        upper_count = int((draws >= 0).sum())
        # Davison-Hinkley style finite Monte-Carlo correction prevents p=0.
        p_value = min(1.0, 2.0 * min((lower_count + 1) / (RESAMPLES + 1), (upper_count + 1) / (RESAMPLES + 1)))
        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "videos": len(differences),
                "mean_recall_difference_a_minus_b": float(differences.mean()),
                "bootstrap95_low": float(np.quantile(draws, 0.025)),
                "bootstrap95_high": float(np.quantile(draws, 0.975)),
                "two_sided_bootstrap_p_plus_one": p_value,
                "resamples": RESAMPLES,
            }
        )
    adjusted = holm([row["two_sided_bootstrap_p_plus_one"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["holm_adjusted_p_plus_one"] = value
    pd.DataFrame(rows).to_csv(OUT / "paired_video_comparisons.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "protocol": "ard100-det-supp-v2",
        "status": "PASS",
        "metric": "video-macro top-1 IoU50 recall",
        "resamples": RESAMPLES,
        "finite_resample_correction": "plus one in each empirical tail",
        "multiple_comparison_control": "Holm over 15 model pairs",
        "input_hashes": hashes,
        "output": "paired_video_comparisons.csv",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
