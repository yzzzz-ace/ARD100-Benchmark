# Supplementary reproducibility record

This record belongs to protocol `ard100-det-v1` plus the preregistered
validation-only supplement `ard100-det-supp-v2`. The official-test predictions,
weights, thresholds, and manifests are immutable.

## Data and evaluation identity

- Source membership: 65 learned videos and 35 official-test videos.
- Learned split: 52 training videos (35,708 frames) and 13 validation videos
  (8,938 frames), with no parent-video overlap.
- Official test: 72,631 annotated frames; 71,633 target-present and 998
  explicit target-absent frames. The absent frames occur in 15 videos.
- Primary input: 640 pixels; paired inference-only diagnostic: 1280 pixels.
- AP: COCO 101-point AP, confidence floor 0.001.
- Operating threshold: validation-only top-1 IoU50 F1, supplemented by
  validation-only absent-FPR constraints of 0.05, 0.10, and 0.20.
- Statistical unit for headline recall: video. Paired uncertainty uses 10,000
  video bootstrap resamples, plus-one empirical p-values, and Holm correction
  over all 15 model pairs.

## Primary training recipes

| Model | Initialization | Optimizer and batch | Completed / selected epoch | Parameters |
|---|---|---:|---:|---:|
| Faster R-CNN R50-FPN v2 | public COCO checkpoint | Nesterov SGD, batch 8 | 12 / 4 | 43.26 M |
| RetinaNet R50-FPN v2 | public COCO checkpoint | Nesterov SGD, batch 8 | 30 / 26 | 36.33 M |
| FCOS R50-FPN | public COCO checkpoint | Nesterov SGD, batch 8 | 12 / 2 | 32.12 M |
| YOLOv8s | public COCO checkpoint | SGD, batch 64 | 18 / 9 | 11.14 M |
| RT-DETR-L | public COCO checkpoint | AdamW, batch 16 | 17 / 8 | 32.81 M |
| YOLOMG-Arch | compatible 32/510 tensors from released YOLOv5s initialization | released SGD hyperparameters, batch 8 | 30 / 26 | 2.04 M |

Torchvision uses learning rate 0.0025, momentum 0.9, weight decay
`5e-4`, Nesterov momentum, and cosine decay. YOLOv8s uses initial learning rate
0.01, momentum 0.937, weight decay `5e-4`, and three warm-up epochs. RT-DETR-L
uses initial learning rate `1e-4`, momentum 0.9, weight decay `1e-4`, and zero
bias-group warm-up; a generic 0.1 bias warm-up was excluded after producing
non-finite loss. The primary RGB runs use seed 20260821. The released YOLOMG
stack uses seed 1, which is disclosed and tested by validation-only additional
seeds rather than by reopening the official test.

The active benchmark environment records Ultralytics 8.4.67, PyTorch
2.11.0+cu126, and Torchvision 0.26.0+cu126. Individual supplemental run
manifests also record Python, CUDA, GPU identity, initialization hashes, and
complete training arguments.

## Runtime boundary

All detector latency results are FP32, batch one, on one RTX 4090, after 100
warm-up iterations and over 1,000 already decoded BGR frames. Timing ends after
final predictions. YOLOMG receives precomputed motion at this boundary.
Separately, generation of 117,277 motion inputs took 22,211.2 seconds, or
189.4 ms/frame (5.28 frames/s), including disk I/O, resize, Lucas--Kanade
tracking, RANSAC alignment, and image writing. The two throughput values are not
interchangeable.

## Supplemental analysis status

| Item | Output | Status |
|---|---|---|
| Corrected paired statistics | `results/statistics/` | PASS |
| Fixed-FPR operating points and FROC | `results/operating_points/` | PASS |
| Adjusted attribute associations | `results/multivariable/` | PASS |
| Deterministic qualitative cases | `figures/qualitative_main_manifest.json` | PASS |
| Random-frame split construction | `manifests/random_frame_split_manifest.json` | PASS |
| Random-frame YOLOv8 diagnostic | `results/robustness/leakage_diagnostic.json` | PASS |
| Three-seed validation stability | `results/robustness/` | PASS |

The non-convergent first GEE implementation is excluded from the public result
set and manuscript. The released fixed-video-effect binomial GLM outputs are
under `results/multivariable/` with unambiguous filenames.

## Validation-only split and seed diagnostics

The matched random-frame YOLOv8s run reaches native validation AP50:95 of
0.39773, compared with 0.09702 for the video-disjoint run. The difference is
0.30071 absolute, or 310% relative inflation from the video-disjoint value.
All 65 learned videos occur in both random-frame subsets, and 30.9% of adjacent
sampled-frame pairs cross the split. This is a leakage diagnostic, not an
official-test result.

Three-seed summaries on the fixed video-disjoint validation set are:

| Model | AP50:95 mean +/- SD | Macro recall mean +/- SD | AP range |
|---|---:|---:|---:|
| YOLOv8s | 0.1114 +/- 0.0102 | 0.2519 +/- 0.0053 | 0.1011--0.1215 |
| RT-DETR-L | 0.2661 +/- 0.0030 | 0.5125 +/- 0.0033 | 0.2627--0.2686 |
| YOLOMG-Arch | 0.3104 +/- 0.0048 | 0.6756 +/- 0.0251 | 0.3060--0.3155 |

The AP and macro-recall ordering is YOLOMG-Arch, RT-DETR-L, then YOLOv8s in
every seed. Supplemental completion is recorded by the PASS manifest in
`manifests/supplement_training_manifest.json`; the official test was not
reopened.
