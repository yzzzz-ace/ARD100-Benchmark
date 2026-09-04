#!/usr/bin/env python3
"""Build a source-faithful YOLOMG motion branch on the frozen video split.

The motion image follows the released paper implementation: a central frame is
paired with frames at t-2 and t+2; both neighbors are aligned to the central
frame by grid LK flow and a RANSAC homography; aligned absolute differences are
averaged in floating point. RGB images and labels are hard-linked from the
already audited benchmark sources.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from public_paths import DATA_ROOT, OFFICIAL_ROOT, REPO_ROOT, YOLOMG_CODE_ROOT


ROOT = REPO_ROOT
GLOBAL_SOURCE = DATA_ROOT
FULLFRAME_SOURCE = ROOT / "datasets" / "ARD100_FullFrame_v1"
LINEAGE = OFFICIAL_ROOT / "manifests" / "dataset_lineage.csv"
OUTPUT = ROOT / "datasets" / "ARD100_YOLOMG_Paper_v1"
REFERENCE_FD = YOLOMG_CODE_ROOT / "test_code" / "FD5_mask.py"
REFERENCE_ALIGNMENT = YOLOMG_CODE_ROOT / "test_code" / "MOD_Functions.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resize_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale == 1.0:
        return image
    return cv2.resize(image, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)


def hardlink_or_verify(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise ValueError(f"Existing hard-link target has wrong size: {destination}")
        return
    os.link(source, destination)


def official_video_map() -> dict[str, Path]:
    with LINEAGE.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    mapping = {row["official_id"]: Path(row["official_video"]) for row in rows}
    if len(mapping) != 100:
        raise ValueError(f"Expected 100 official videos, found {len(mapping)}")
    return mapping


def source_roots(split: str) -> tuple[Path, Path]:
    if split in {"train", "val"}:
        return GLOBAL_SOURCE / "images" / split, GLOBAL_SOURCE / "labels" / split
    return FULLFRAME_SOURCE / "images" / "test", FULLFRAME_SOURCE / "labels" / "test"


def prepare_links() -> dict[str, dict[int, tuple[str, str]]]:
    needed: dict[str, dict[int, tuple[str, str]]] = defaultdict(dict)
    split_root = OUTPUT / "splits"
    split_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        rgb_source, label_source = source_roots(split)
        rgb_paths = sorted(rgb_source.glob("*.jpg"))
        if not rgb_paths:
            raise FileNotFoundError(rgb_source)
        rgb_lines: list[str] = []
        motion_lines: list[str] = []
        for source_rgb in rgb_paths:
            sample_id = source_rgb.stem
            video_id, frame_text = sample_id.rsplit("_", 1)
            frame_index = int(frame_text)
            if frame_index in needed[video_id]:
                raise ValueError(f"Duplicate requested frame: {sample_id}")
            needed[video_id][frame_index] = (split, sample_id)
            destination_rgb = OUTPUT / "images" / split / source_rgb.name
            destination_label = OUTPUT / "labels" / split / f"{sample_id}.txt"
            destination_motion = OUTPUT / "images2" / split / source_rgb.name
            hardlink_or_verify(source_rgb, destination_rgb)
            hardlink_or_verify(label_source / f"{sample_id}.txt", destination_label)
            rgb_lines.append(destination_rgb.as_posix())
            motion_lines.append(destination_motion.as_posix())
        (split_root / f"{split}_rgb.txt").write_text("\n".join(rgb_lines) + "\n", encoding="utf-8")
        (split_root / f"{split}_motion.txt").write_text("\n".join(motion_lines) + "\n", encoding="utf-8")
    return needed


def grid_points(width: int, height: int) -> np.ndarray:
    scale = 2
    grid_width = 32 * scale
    grid_height = 24 * scale
    columns = int(width / grid_width - 1)
    rows = int(height / grid_height - 1)
    points = [
        (np.float32(i * grid_width + grid_width / 2.0), np.float32(j * grid_height + grid_height / 2.0))
        for i in range(columns)
        for j in range(rows)
    ]
    return np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)


_POINT_CACHE: dict[tuple[int, int], np.ndarray] = {}


def motion_compensate(neighbor: np.ndarray, central: np.ndarray) -> tuple[np.ndarray, bool]:
    height, width = central.shape
    resized_width, resized_height = 1920, 1080
    neighbor_grid = cv2.resize(neighbor, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
    central_grid = cv2.resize(central, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
    points = _POINT_CACHE.setdefault((resized_width, resized_height), grid_points(resized_width, resized_height))
    lk = dict(
        winSize=(15, 15),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.003),
    )
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(neighbor_grid, central_grid, points, None, **lk)
    fallback = tracked is None or status is None
    if not fallback:
        good_new = tracked[status.reshape(-1) == 1]
        good_old = points[status.reshape(-1) == 1]
        fallback = len(good_old) < 15
    if fallback:
        homography = np.eye(3, dtype=np.float64)
    else:
        homography, _ = cv2.findHomography(good_new, good_old, cv2.RANSAC, 3.0)
        if homography is None or not np.isfinite(homography).all():
            homography = np.eye(3, dtype=np.float64)
            fallback = True
    compensated = cv2.warpPerspective(
        neighbor,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
    )
    return compensated, fallback


def paper_motion(previous2: np.ndarray, central: np.ndarray, future2: np.ndarray) -> tuple[np.ndarray, int]:
    grays = [cv2.cvtColor(cv2.GaussianBlur(frame, (11, 11), 0), cv2.COLOR_BGR2GRAY) for frame in (previous2, central, future2)]
    aligned_previous, fallback_previous = motion_compensate(grays[0], grays[1])
    aligned_future, fallback_future = motion_compensate(grays[2], grays[1])
    difference_previous = cv2.absdiff(grays[1], aligned_previous).astype(np.float32)
    difference_future = cv2.absdiff(grays[1], aligned_future).astype(np.float32)
    difference = np.clip((difference_previous + difference_future) / 2.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(difference, cv2.COLOR_GRAY2BGR), int(fallback_previous) + int(fallback_future)


def write_motion(path: Path, image: np.ndarray, max_side: int, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    resized = resize_max_side(image, max_side)
    if not cv2.imwrite(str(path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), quality]):
        raise OSError(path)


def build_motion(needed: dict[str, dict[int, tuple[str, str]]], max_side: int, quality: int) -> dict[str, object]:
    videos = official_video_map()
    generated = skipped = zero_boundary = alignment_fallbacks = 0
    per_video: list[dict[str, object]] = []
    for video_index, video_id in enumerate(sorted(needed), start=1):
        requests = needed[video_id]
        missing = {
            frame_index
            for frame_index, (split, sample_id) in requests.items()
            if not (OUTPUT / "images2" / split / f"{sample_id}.jpg").exists()
        }
        skipped += len(requests) - len(missing)
        if not missing:
            continue
        capture = cv2.VideoCapture(str(videos[video_id]))
        if not capture.isOpened():
            raise OSError(videos[video_id])
        window: deque[tuple[int, np.ndarray]] = deque(maxlen=5)
        decoded = 0
        built_here = 0
        started = time.time()
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                decoded += 1
                window.append((decoded, frame))
                if len(window) < 5:
                    continue
                central_index = window[2][0]
                if central_index not in missing:
                    continue
                split, sample_id = requests[central_index]
                motion, fallbacks = paper_motion(window[0][1], window[2][1], window[4][1])
                alignment_fallbacks += fallbacks
                write_motion(OUTPUT / "images2" / split / f"{sample_id}.jpg", motion, max_side, quality)
                missing.remove(central_index)
                generated += 1
                built_here += 1
        finally:
            capture.release()

        # The released t-2/t+2 formulation is undefined on the first and last
        # two frames. Preserve the common all-frame denominator with explicit
        # zero motion for these boundary samples.
        valid_boundary_indices = {1, 2, max(decoded - 1, 1), decoded}
        unexpected_missing = missing - valid_boundary_indices
        if unexpected_missing:
            raise RuntimeError(
                f"Non-boundary motion samples were not generated for {video_id}: "
                f"{sorted(unexpected_missing)[:10]}"
            )
        for frame_index in sorted(missing):
            split, sample_id = requests[frame_index]
            rgb_path = OUTPUT / "images" / split / f"{sample_id}.jpg"
            image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if image is None:
                raise OSError(rgb_path)
            write_motion(OUTPUT / "images2" / split / f"{sample_id}.jpg", np.zeros_like(image), max_side, quality)
            generated += 1
            built_here += 1
            zero_boundary += 1
        per_video.append(
            {
                "video_id": video_id,
                "decoded_frames": decoded,
                "requested_samples": len(requests),
                "generated_now": built_here,
                "seconds": time.time() - started,
            }
        )
        print(
            f"[PAPER-MOTION] {video_index}/{len(needed)} {video_id}: "
            f"decoded={decoded} requested={len(requests)} built={built_here} seconds={time.time() - started:.1f}",
            flush=True,
        )
    return {
        "generated_now": generated,
        "already_present": skipped,
        "zero_boundary_frames": zero_boundary,
        "alignment_fallbacks": alignment_fallbacks,
        "per_video": per_video,
    }


def audit_and_write_config(build_summary: dict[str, object]) -> dict[str, object]:
    counts: dict[str, int] = {}
    errors: list[str] = []
    for split in ("train", "val", "test"):
        stems = {
            branch: {path.stem for path in (OUTPUT / branch / split).glob("*.jpg")}
            for branch in ("images", "images2")
        }
        label_stems = {path.stem for path in (OUTPUT / "labels" / split).glob("*.txt")}
        counts[split] = len(stems["images"])
        if stems["images"] != stems["images2"] or stems["images"] != label_stems:
            errors.append(f"{split}_stem_mismatch")
    expected = {"train": 35708, "val": 8938, "test": 72631}
    for split, value in expected.items():
        if counts.get(split) != value:
            errors.append(f"{split}_count={counts.get(split)} expected={value}")

    split_root = OUTPUT / "splits"
    yaml_text = (
        f"train: {(split_root / 'train_rgb.txt').as_posix()}\n"
        f"train02: {(split_root / 'train_motion.txt').as_posix()}\n"
        f"val: {(split_root / 'val_rgb.txt').as_posix()}\n"
        f"val2: {(split_root / 'val_motion.txt').as_posix()}\n"
        f"test: {(split_root / 'test_rgb.txt').as_posix()}\n"
        f"test2: {(split_root / 'test_motion.txt').as_posix()}\n"
        "nc: 1\n"
        "names: ['Drone']\n"
    )
    (OUTPUT / "dataset.yaml").write_text(yaml_text, encoding="utf-8")
    manifest = {
        "protocol_version": "ard100-det-v1",
        "variant": "YOLOMG paper-motion branch",
        "temporal_context": "t-2, t, t+2",
        "alignment": "released grid LK plus RANSAC homography",
        "difference": "floating-point mean of two aligned absolute differences",
        "boundary_policy": "zero motion where t-2 or t+2 is unavailable",
        "storage_max_side": 1280,
        "counts": counts,
        "reference_fd5_sha256": sha256(REFERENCE_FD),
        "reference_alignment_sha256": sha256(REFERENCE_ALIGNMENT),
        "lineage_sha256": sha256(LINEAGE),
        "build": build_summary,
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }
    (OUTPUT / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if errors:
        raise RuntimeError(errors)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-side", type=int, default=1280)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--links-only", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    needed = prepare_links()
    summary = {"generated_now": 0, "already_present": 0, "zero_boundary_frames": 0, "alignment_fallbacks": 0}
    if not args.links_only:
        summary = build_motion(needed, args.max_side, args.jpeg_quality)
    manifest = audit_and_write_config(summary)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
