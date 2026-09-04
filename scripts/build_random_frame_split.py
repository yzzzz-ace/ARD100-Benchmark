#!/usr/bin/env python3
"""Build the preregistered random-frame leakage diagnostic split."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from public_paths import DATA_ROOT, REPO_ROOT


ROOT = REPO_ROOT
SOURCE = DATA_ROOT
OUT = ROOT / "datasets" / "ARD100_RandomFrameSplit_v2"
SEED = 20260821
TRAIN_COUNT = 35708
VAL_COUNT = 8938


def video_id(path: Path) -> str:
    return path.stem.rsplit("_", 1)[0]


def frame_index(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def score(path: Path) -> str:
    return hashlib.sha256(f"{SEED}:{path.stem}".encode()).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    images = sorted((SOURCE / "images" / "train").glob("*.jpg")) + sorted((SOURCE / "images" / "val").glob("*.jpg"))
    if len(images) != TRAIN_COUNT + VAL_COUNT:
        raise ValueError(f"Expected {TRAIN_COUNT + VAL_COUNT} images, found {len(images)}")
    ranked = sorted(images, key=lambda path: (score(path), path.stem))
    val_set = set(ranked[:VAL_COUNT])
    train = sorted([path for path in images if path not in val_set])
    val = sorted(val_set)
    train_videos = {video_id(path) for path in train}
    val_videos = {video_id(path) for path in val}
    all_videos = train_videos | val_videos
    if len(train) != TRAIN_COUNT or len(val) != VAL_COUNT or len(all_videos) != 65:
        raise AssertionError("Random split denominator mismatch")
    if train_videos != all_videos or val_videos != all_videos:
        raise AssertionError("Every source video must occur in both random-frame partitions")

    OUT.mkdir(parents=True)
    train_list = OUT / "train.txt"
    val_list = OUT / "val.txt"
    train_list.write_text("\n".join(str(path.resolve()) for path in train) + "\n", encoding="utf-8")
    val_list.write_text("\n".join(str(path.resolve()) for path in val) + "\n", encoding="utf-8")
    train_ids = OUT / "train_sample_ids.txt"
    val_ids = OUT / "val_sample_ids.txt"
    train_ids.write_text("\n".join(path.stem for path in train) + "\n", encoding="utf-8")
    val_ids.write_text("\n".join(path.stem for path in val) + "\n", encoding="utf-8")
    yaml_path = OUT / "dataset.yaml"
    yaml_path.write_text(
        f"path: {OUT.as_posix()}\ntrain: {train_list.as_posix()}\nval: {val_list.as_posix()}\n"
        "nc: 1\nnames:\n  0: drone\n",
        encoding="utf-8",
    )
    assignments = {path.stem: "val" if path in val_set else "train" for path in images}
    by_video = defaultdict(list)
    for path in images:
        by_video[video_id(path)].append(path)
    adjacent_cross_pairs = 0
    adjacent_pairs = 0
    per_video = []
    for name, paths in sorted(by_video.items()):
        ordered = sorted(paths, key=frame_index)
        cross = total = 0
        for one, two in zip(ordered, ordered[1:]):
            if frame_index(two) - frame_index(one) == 1:
                total += 1
                cross += int(assignments[one.stem] != assignments[two.stem])
        adjacent_cross_pairs += cross
        adjacent_pairs += total
        per_video.append(
            {
                "video_id": name,
                "train_frames": sum(path not in val_set for path in paths),
                "val_frames": sum(path in val_set for path in paths),
                "adjacent_pairs": total,
                "adjacent_cross_partition_pairs": cross,
            }
        )
    manifest = {
        "protocol": "ard100-det-supp-v2",
        "status": "PASS",
        "seed": SEED,
        "method": "global SHA256 ranking of sample IDs",
        "counts": {"train": len(train), "val": len(val), "videos": len(all_videos)},
        "video_overlap": len(train_videos & val_videos),
        "adjacent_pairs": adjacent_pairs,
        "adjacent_cross_partition_pairs": adjacent_cross_pairs,
        "adjacent_cross_partition_rate": adjacent_cross_pairs / max(adjacent_pairs, 1),
        "files": {
            "train": {"path": str(train_list), "sha256": sha256(train_list)},
            "val": {"path": str(val_list), "sha256": sha256(val_list)},
            "train_sample_ids": {"path": str(train_ids), "sha256": sha256(train_ids)},
            "val_sample_ids": {"path": str(val_ids), "sha256": sha256(val_ids)},
            "yaml": {"path": str(yaml_path), "sha256": sha256(yaml_path)},
        },
        "per_video": per_video,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "per_video"}, indent=2))


if __name__ == "__main__":
    main()
