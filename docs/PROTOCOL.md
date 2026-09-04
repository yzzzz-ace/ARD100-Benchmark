# ARD100 video-disjoint detector benchmark protocol

Protocol identifier: `ard100-det-v1`  
Primary-test status: frozen and complete

## Research questions

- **RQ1 — Architecture:** How do representative two-stage, dense one-stage,
  real-time one-stage, transformer, and compact two-stream detectors generalize
  to previously unseen ARD100 videos?
- **RQ2 — Operating point:** Does detector ranking remain stable under
  validation-F1 and validation-constrained false-positive-rate thresholds?
- **RQ3 — Failure factors:** Which observable target and scene attributes are
  associated with localization failure?
- **RQ4 — Resolution and efficiency:** What accuracy/latency trade-offs arise
  at 640 and 1280 pixels under a common inference boundary?
- **RQ5 — Evaluation leakage:** How much can random frame splitting inflate
  validation accuracy relative to source-video-disjoint splitting?

## Data partition

- The 65 learned videos are split into 52 training and 13 validation videos.
- The official 35-video ARD100 test set is held out by parent video.
- No frame, parent video, checkpoint selection, confidence threshold, or
  hyperparameter from the official test enters training or validation.
- Exact video identities are stored under `splits/`.
- A separate random-frame YOLOv8s run is a validation-only leakage diagnostic;
  it is never evaluated on the official test.

## Primary detector set

The fixed comparison contains six detectors:

1. Faster R-CNN R50-FPN v2;
2. RetinaNet R50-FPN v2;
3. FCOS R50-FPN;
4. YOLOv8s;
5. RT-DETR-L;
6. YOLOMG-Arch, the compact two-stream architecture and motion construction
   reproduced from the public YOLOMG implementation.

All models solve the same single-class full-frame detection problem. RGB-only
models receive identical RGB frames and boxes. YOLOMG-Arch additionally uses
its published motion input; boundary frames use explicit zero-motion padding,
so every detector is evaluated on the same frame denominator.

## Training and selection

- Public COCO initialization is used when supported. YOLOMG-Arch follows the
  compatible partial transfer defined by its released implementation.
- Training is capped at 30 epochs with validation-only checkpoint selection.
- The primary input long side is 640 pixels.
- The original primary seed is retained. Additional validation-only seeds are
  reported for YOLOv8s, RT-DETR-L, and YOLOMG-Arch.
- Test scores are never used to replace a checkpoint or tune a threshold.

Full optimizer and software details are recorded in
`docs/REPRODUCIBILITY.md` and the supplied result manifests.

## Evaluation

- COCO AP@[.50:.95] and AP50 use a confidence floor of 0.001.
- The headline threshold is selected on validation predictions by top-1
  IoU50 F1.
- Additional operating points use validation absent-frame FPR constraints of
  0.05, 0.10, and 0.20.
- Recall is reported both per frame and as an unweighted mean across videos.
- Target-absent performance is reported explicitly.
- Paired uncertainty resamples the 35 test videos 10,000 times; pairwise
  p-values use a plus-one correction and Holm family-wise adjustment.

## Resolution and runtime

- Training and the primary locked evaluation use 640 pixels.
- The 1280-pixel condition changes inference resolution only and is reported as
  a diagnostic, not a second tuned leaderboard.
- Runtime is FP32, batch one, on one RTX 4090 after 100 warm-up iterations and
  over 1,000 already decoded BGR frames.
- Timing ends after final detections. YOLOMG-Arch receives precomputed motion at
  the common model-only boundary; its motion-construction cost is disclosed
  separately and must not be conflated with detector-only latency.

## Factor and qualitative analyses

Precomputed target size, contrast, local texture/blur, surrounding edge
density, target displacement, and camera displacement support descriptive
strata and fixed-video-effect binomial models. Coefficients are interpreted as
associations rather than causal effects. Qualitative examples are selected by
deterministic disagreement rules documented in the supplement.

## Freeze rule

The archived lock digest and `manifests/locked_test_manifest.json` identify the
one-time official-test suite. Reproductions must write to a new output root and
must not overwrite the archived result tables, manifests, or checkpoint
digests distributed with this package.
