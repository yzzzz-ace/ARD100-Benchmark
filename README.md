# ARD100 Benchmark

Official release package for the study:

> **Tiny Object Detection in Long-Range Videos: A Video-Disjoint and
> Attribute-Stratified Benchmark on ARD100**

This repository evaluates six detector families on ARD100 under a restored
65/35-video source partition, a video-disjoint 52/13-video learned split,
validation-only model selection, and a locked one-time official test. It is
independent of LCA-DR and does not contain a tracker.

## Main findings

- At 640 input, YOLOMG-Arch reaches AP50:95 0.3189 and video-macro IoU50
  recall 0.7397; RT-DETR-L is the strongest RGB-only detector at 0.1793 and
  0.3526.
- A matched random-frame split raises YOLOv8s validation AP50:95 from 0.0970
  to 0.3977, an absolute increase of 0.3007 and 310% relative inflation.
- Across three validation seeds, the YOLOMG-Arch, RT-DETR-L, YOLOv8s ordering
  is unchanged.
- The reported 79.2 FPS for YOLOMG-Arch is model-only. Recorded offline motion
  construction averaged 189.4 ms/frame (5.28 frames/s) including I/O.

## Repository contents

| Path | Content |
|---|---|
| `scripts/` | Dataset construction, training, inference, evaluation, statistics, and figure generation |
| `configs/` | Portable benchmark and dataset templates |
| `splits/` | Official/video-disjoint memberships and exact random-frame diagnostic sample IDs |
| `results/` | Compact machine-readable results used by the paper |
| `manifests/` | Dataset, frozen-test, and supplemental audit records |
| `figures/` | Final benchmark and diagnostic figures |
| `docs/` | Frozen protocols, data setup, and reproducibility details |
| `weights/` | Checksum table and release-asset instructions; no checkpoints are committed to Git |

The ARD100 images, annotations, videos, generated motion inputs, full
prediction CSVs, training caches, and optimizer states are intentionally not
included.

## Data setup

Download ARD100 from the source authors' release and place or link the prepared
YOLO dataset outside this repository. Set the following environment variables:

```powershell
$env:ARD100_DATASET_ROOT = 'D:\data\ARD100_prepared'
$env:ARD100_OFFICIAL_ROOT = 'D:\data\ARD100_official_split'
$env:YOLOMG_CODE_ROOT = 'D:\code\YOLOMG'
```

The expected prepared dataset layout is:

```text
ARD100_prepared/
  images/{train,val,test}/
  labels/{train,val,test}/
```

See `docs/DATA_SETUP.md` for the source link, split reconstruction, annotation
rules, and motion-input construction.

## Environment

The recorded benchmark environment used Python with PyTorch 2.11.0+cu126,
Torchvision 0.26.0+cu126, and Ultralytics 8.4.67 on one RTX 4090. Install the
analysis dependencies with:

```bash
pip install -r requirements.txt
```

Install the CUDA-compatible PyTorch build using the official PyTorch index for
your platform before reproducing GPU experiments.

## Reproduction entry points

Examples below are run from the repository root.

```bash
# Inspect all command-line options
python scripts/train_torchvision_baseline.py --help
python scripts/train_ultralytics_baseline.py --help
python scripts/train_yolomg_paper.py --help

# Recompute compact analyses from generated predictions
python scripts/compare_models.py --help
python scripts/reanalyze_statistics_v2.py
python scripts/analyze_operating_points.py
python scripts/analyze_multivariable.py

# Check the public release package before publication
python scripts/verify_release.py
```

The immutable official test was run once in the original study. The public
lock and result manifests are supplied for audit; users reproducing the study
should create a fresh local lock rather than overwrite the archived result.

## Checkpoints

Six selected checkpoints total approximately 518.4 MB and should be attached
to a GitHub Release or deposited in a DOI-bearing archive. Their filenames and
SHA-256 digests are listed in `weights/README.md`. Checkpoints are optional for
using the analysis results but required to reproduce inference.

## Data and code availability

ARD100 is third-party research data. This repository provides split identities,
construction scripts, hashes, and derived aggregate results, but does not
redistribute the source videos or images. Users must obtain the dataset from
the source authors and comply with its terms.

The benchmark harness is released under the MIT License. The external YOLOMG
repository is GPL-3.0 licensed and is not vendored here. See
`THIRD_PARTY_NOTICES.md` before distributing modified third-party files.

## Citation

Author metadata and the planned repository URL are recorded in `CITATION.cff`.
Until the article receives a DOI, cite the tagged repository release and the
original ARD100/YOLOMG paper.
