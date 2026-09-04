# ARD100 benchmark supplemental protocol v2

Status: preregistered before supplemental analysis/training.

The completed `ard100-det-v1` official-test predictions, metrics, weights,
validation thresholds, `configs/test_lock.json`, and
`manifests/locked_test_manifest.json` are immutable inputs. No detector is
retrained for a replacement official-test score and no official-test inference
is repeated.

## S1. Corrected paired uncertainty

- Reuse the 35 per-video recall values already produced by the locked test.
- Use 10,000 paired video bootstrap resamples.
- Estimate a two-sided empirical p-value with a plus-one correction in both
  tails, then apply Holm family-wise correction to all 15 model pairs.
- The released corrected pairwise table is
  `results/statistics/corrected_pairwise_comparisons.csv`; do not overwrite
  the original comparison tables.

## S2. Operating-point analysis

- Candidate thresholds are derived from validation top-1 scores only.
- In addition to the pre-existing validation-F1 operating point, select the
  lowest validation threshold whose absent-frame FPR does not exceed 0.05,
  0.10, or 0.20; ties favor higher localization recall and then precision.
- Apply each frozen threshold once to the already stored test predictions.
- Full test threshold sweeps are descriptive FROC curves, not model-selection
  procedures.
- Bootstrap absent-frame FPR by resampling the 15 test videos that contain
  explicit absent frames.

## S3. Attribute associations

- The primary factor bins in `results/factors/` remain the predeclared
  descriptive analysis.
- A secondary binomial generalized linear model uses IoU50 localization as the
  response, standardized precomputed attributes as predictors, fixed effects
  for test video, and video-clustered covariance. The preregistered GEE fit is
  retained as a convergence audit but excluded because five of six fits did
  not converge.
- Report odds ratios and cluster-robust 95% intervals. Interpret associations,
  not causal effects.

## S4. Qualitative panels

- Select four cases by deterministic rules recorded in the figure manifest:
  tiny/low-contrast, high-camera-motion, near-stationary, and target-absent.
- Within each eligible stratum, select maximum six-model disagreement; the
  absent case selects the highest count of locked above-threshold responses.
- Resolve ties lexicographically by sample ID.

## S5. Training-seed stability

- Models: YOLOv8s, RT-DETR-L, and the common-budget YOLOMG architecture/motion
  implementation.
- Original seeds are retained as one replicate. Add seeds 20260822 and 20260823
  for Ultralytics models and seeds 2 and 3 for YOLOMG.
- Keep model initialization, 640 input, batch size, optimizer, augmentation,
  maximum 30 epochs, validation split, and early stopping identical to the
  primary run.
- Run validation inference only. Report common-evaluator validation AP50:95 and
  macro video recall as mean, standard deviation, and range across three seeds.
- Supplemental seed runs never enter the official-test leaderboard. Compact
  summaries are under `results/robustness/`, with validation metrics under
  `results/metrics/`.

## S6. Frame-split leakage diagnostic

- Detector: YOLOv8s only, as a computationally economical diagnostic.
- Pool all 44,646 learned-split frames from the same 65 source training videos.
- Use a deterministic hash-randomized 35,708/8,938 train/validation frame split.
- Require all source videos to appear in both learned partitions and record the
  number of adjacent cross-partition frame pairs.
- Train with the identical YOLOv8s recipe and seed 20260821.
- Compare its best native validation AP50:95 with the original video-disjoint
  run. This quantifies validation inflation only and is never tested on the
  official 35 videos. Exact diagnostic sample IDs are released under
  `splits/random_frame_*_sample_ids.txt`.

## S7. Motion-construction cost

- Reuse the completed dataset-build manifest: 117,277 motion images, 22,211.2 s
  total construction time, approximately 5.28 frames/s including disk I/O,
  resizing, Lucas-Kanade tracking, RANSAC alignment, and image writing.
- Keep this boundary separate from the batch-one model-only benchmark.
