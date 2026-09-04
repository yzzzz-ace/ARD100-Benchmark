# Data setup

## Source data

ARD100 is distributed by the YOLOMG authors. At the time this release package
was prepared, the source repository README linked the dataset through Baidu
Cloud:

`https://github.com/Irisky123/YOLOMG`

`https://pan.baidu.com/s/1ycAoKbzQ1rlzvKr8VRakgw?pwd=1x2z`

This repository does not redistribute the videos, extracted frames,
annotations, or generated motion inputs. Verify the current source terms before
redistributing any third-party data.

## Frozen memberships

- Source partition: 65 learned videos and 35 official-test videos.
- Learned partition: 52 training videos and 13 validation videos.
- No video belongs to more than one learned or official partition.
- Frames without matching XML are omitted rather than converted to negatives.
- XML files explicitly containing no target remain valid negative frames.

Exact memberships are in `splits/`. The matched random-frame diagnostic is
generated deterministically by SHA-256 ranking of sample IDs with seed
20260821; its audited counts are 35,708 training and 8,938 validation frames.

## Prepared layout

Set `ARD100_DATASET_ROOT` to a YOLO-style dataset with:

```text
images/train, images/val, images/test
labels/train, labels/val, labels/test
```

Set `ARD100_OFFICIAL_ROOT` when rebuilding full-frame test material from source
videos and XML. Set `YOLOMG_CODE_ROOT` to a clone of the released GPL-3.0
YOLOMG repository for the dual-stream model and motion preprocessing.

Dataset construction is intentionally idempotent: an existing output is
verified and never silently overwritten. Keep the source videos outside Git.
