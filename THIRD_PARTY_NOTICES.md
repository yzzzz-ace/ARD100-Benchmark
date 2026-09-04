# Third-party notices

This repository contains an original benchmark harness and does not vendor the
ARD100 media or the YOLOMG source tree.

## ARD100 and YOLOMG

ARD100 and the released YOLOMG implementation accompany the paper *YOLOMG:
Vision-based Drone-to-Drone Detection with Appearance and Pixel-Level Motion
Fusion*. The source repository carries the GNU General Public License version
3. Obtain the dataset and source code from the original authors. Do not assume
that the software licence independently grants permission to redistribute the
dataset media.

- Official source and dataset page: <https://github.com/Irisky123/YOLOMG>
- Paper/preprint: <https://arxiv.org/abs/2503.07115>

The scripts in this repository call an externally installed YOLOMG tree via
`YOLOMG_CODE_ROOT`. If third-party YOLOMG files are copied or modified for a
redistribution, retain the GPL-3.0 notices and comply with that licence.

## Frameworks and pretrained initialization

Torchvision, PyTorch, Ultralytics, OpenCV, NumPy, pandas, Matplotlib, seaborn,
Statsmodels, and Pillow retain their respective licences. Public COCO
initialization checkpoints retain their upstream terms. This repository's MIT
licence applies only to the original benchmark harness and documentation.
