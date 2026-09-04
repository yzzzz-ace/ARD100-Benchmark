#!/usr/bin/env python3
"""Train the source-faithful YOLOMG branch on the frozen learned split."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from public_paths import REPO_ROOT, YOLOMG_CODE_ROOT


ROOT = REPO_ROOT
# Use the released YOLOMG repository root so its legacy absolute-style
# `models` and `utils` imports resolve exactly as intended by train.py.
CODE = YOLOMG_CODE_ROOT
TRAIN_SCRIPT = CODE / "train.py"
DATASET = ROOT / "datasets" / "ARD100_YOLOMG_Paper_v1"
DATA_YAML = DATASET / "dataset.yaml"
MODEL_YAML = CODE / "models" / "NPS_uav_s.yaml"
INITIALIZATION = CODE / "yolov5s.pt"
HYP = CODE / "data" / "hyps" / "hyp.scratch-low.yaml"
PROJECT = ROOT / "outputs" / "training" / "yolomg_paper"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    manifest_path = DATASET / "dataset_manifest.json"
    for path in (TRAIN_SCRIPT, DATA_YAML, MODEL_YAML, INITIALIZATION, HYP, manifest_path):
        if not path.exists():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS":
        raise RuntimeError("YOLOMG paper-motion dataset audit did not pass")

    run_name = f"yolomg_paper_ard100_det_v1_e{args.epochs}_b{args.batch}"
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--weights",
        str(INITIALIZATION),
        "--cfg",
        str(MODEL_YAML),
        "--data",
        str(DATA_YAML),
        "--hyp",
        str(HYP),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch),
        "--imgsz",
        "640",
        "--device",
        "0",
        "--workers",
        str(args.workers),
        "--project",
        str(PROJECT),
        "--name",
        run_name,
        "--optimizer",
        "SGD",
        "--patience",
        "10",
        "--save-period",
        "5",
    ]
    subprocess.run(command, cwd=CODE, check=True)

    run_dir = PROJECT / run_name
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    if not best.exists() or not last.exists():
        raise RuntimeError(f"Missing completed weights in {run_dir}")
    frozen_dir = ROOT / "weights"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    frozen_best = frozen_dir / "yolomg_paper_ard100_det_v1_best.pt"
    frozen_last = frozen_dir / "yolomg_paper_ard100_det_v1_last.pt"
    shutil.copy2(best, frozen_best)
    shutil.copy2(last, frozen_last)
    metadata = {
        "protocol_version": "ard100-det-v1",
        "model": "yolomg_paper",
        "implementation_scope": "released four-head architecture and t-2/t+2 paper-motion preprocessing",
        "epochs": args.epochs,
        "batch": args.batch,
        "seed": 1,
        "seed_note": "released YOLOMG training code calls init_seeds(1 + RANK)",
        "initialization": str(INITIALIZATION),
        "initialization_sha256": sha256(INITIALIZATION),
        "compatible_initialization_tensors_smoke_test": 32,
        "target_state_tensors_smoke_test": 510,
        "model_yaml": str(MODEL_YAML),
        "model_yaml_sha256": sha256(MODEL_YAML),
        "dataset_manifest_sha256": sha256(manifest_path),
        "best": str(frozen_best),
        "best_sha256": sha256(frozen_best),
        "last": str(frozen_last),
        "last_sha256": sha256(frozen_last),
        "command": command,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    (frozen_dir / "yolomg_paper_ard100_det_v1.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
