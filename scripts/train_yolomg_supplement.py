#!/usr/bin/env python3
"""Non-destructive seeded replications of the released YOLOMG training code."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import torch

from public_paths import REPO_ROOT, YOLOMG_CODE_ROOT


ROOT = REPO_ROOT
CODE = YOLOMG_CODE_ROOT
DATA = ROOT / "datasets" / "ARD100_YOLOMG_Paper_v1" / "dataset.yaml"
MODEL_YAML = CODE / "models" / "NPS_uav_s.yaml"
INITIALIZATION = CODE / "yolov5s.pt"
HYP = CODE / "data" / "hyps" / "hyp.scratch-low.yaml"
PROJECT = ROOT / "outputs" / "supplementary" / "training"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tag", default="video_disjoint")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    for path in (CODE / "train.py", DATA, MODEL_YAML, INITIALIZATION, HYP):
        if not path.exists():
            raise FileNotFoundError(path)
    run_name = f"yolomg_arch_{args.tag}_seed{args.seed}"
    run_dir = PROJECT / run_name
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {run_dir}")
    sys.path.insert(0, str(CODE))
    import train as released_train  # noqa: E402
    from utils.general import init_seeds as released_init_seeds  # noqa: E402

    def seeded_init(_ignored: int = 0) -> None:
        released_init_seeds(args.seed)

    released_train.init_seeds = seeded_init
    old_cwd = Path.cwd()
    try:
        import os

        os.chdir(CODE)
        released_train.run(
            weights=str(INITIALIZATION), cfg=str(MODEL_YAML), data=str(DATA), hyp=str(HYP),
            epochs=args.epochs, batch_size=args.batch, imgsz=640, device="0",
            workers=args.workers, project=str(PROJECT), name=run_name,
            optimizer="SGD", patience=10, save_period=5, exist_ok=False,
        )
    finally:
        os.chdir(old_cwd)
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    if not best.exists() or not last.exists():
        raise RuntimeError(f"Incomplete training output in {run_dir}")
    metadata = {
        "protocol": "ard100-det-supp-v2",
        "status": "PASS",
        "model": "yolomg_arch",
        "tag": args.tag,
        "seed": args.seed,
        "seed_injection": "monkeypatch released train.init_seeds without editing released source",
        "data_yaml": str(DATA),
        "data_yaml_sha256": sha256(DATA),
        "model_yaml_sha256": sha256(MODEL_YAML),
        "initialization_sha256": sha256(INITIALIZATION),
        "hyp_sha256": sha256(HYP),
        "epochs_requested": args.epochs,
        "batch": args.batch,
        "best": str(best.resolve()),
        "best_sha256": sha256(best),
        "last_sha256": sha256(last),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    (run_dir / "supplement_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
